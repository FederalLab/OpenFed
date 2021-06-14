import time
from threading import Thread
from typing import Boolean, Callable, Dict

import openfed
from openfed.utils import in_enum
from openfed.utils.safe_exited import safe_exited

from .gpu_info import getGPUInfo
from .informer import Informer
from .sys_info import getSysInfo


class Monitor(Thread):
    """自动启动一个新的线程进行状态的监督。
    这个线程的作用是把当前的系统状态，定期在服务器和客户端之间进行同步。
    """
    informer: Informer
    _hooks_dict: Dict[str, Callable]
    stopped: Boolean

    def __init__(self, informer: Informer, delay: float = 10, auto_start: bool = True):
        """
        Args:
            informer: openfed.federated.Informer
            delay: update frequency.(second)
            auto_start: 是否自动开始执行程序。
                如果是，则自动调用Monitor.start()开始进入后台执行。
                如果不是，则不调用，需要后期手动调用。（如果你想注册一些新的方法hook的时候，显得很有用。）
        """
        super().__init__()
        self.stopped = False

        self._hooks_dict = dict()

        self.delay = delay  # time to collect system information
        self.informer = informer
        if auto_start:
            self.start()

    def register_hook(self, name: str, hook: Callable):
        """
        hook计算出来的数据，将会以name为键值，向外传输。
        hook应该是闭包函数，调用过程不会传入任何参数。

        .. node::
            这里的name，不能和任何现有的系统预置的键值重复！
            这里注册的hook，必须返回一个字典，后者可以被josnize的对象！
        """
        assert not in_enum(
            name, openfed.CONSTANTS), "Invalid name: %s (conflict with default name)" % name

        self._hooks_dict[name] = hook

    def run(self):
        while openfed.ALIVE and not self.stopped:
            if openfed._APPROVED == openfed.APPROVED.ALL:
                self.informer.set_sys_state(getSysInfo())
                self.informer.set_gpu_state(getGPUInfo())
            elif openfed._APPROVED == openfed.APPROVED.SYS:
                self.informer.set_sys_state(getSysInfo())
            elif openfed._APPROVED == openfed.APPROVED.GPU:
                self.informer.set_gpu_state(getGPUInfo())
            else:
                self.informer.set_sys_state({})
                self.informer.set_gpu_state({})

            for name, hook in self._hooks_dict.items():
                self.informer.set(name, hook())

            time.sleep(self.delay)
        else:
            safe_exited()

    def manual_stop(self):
        """Provide a function to end it manually.
        """
        self.stopped = True
