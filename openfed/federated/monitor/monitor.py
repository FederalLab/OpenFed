import time
from collections import OrderedDict
from threading import Thread
from typing import Boolean, Callable, Dict

from openfed.federated.core.federated_c10d import FederatedWorld, Store
from openfed.federated.register import World
from openfed.federated.utils.safe_exited import safe_exited
from openfed.utils.types import STATUS
from .informer import Informer


class Monitor(Thread):
    """自动启动一个新的线程进行状态的监督。
    这个线程的作用是把当前的系统状态，定期在服务器和客户端之间进行同步。
    """
    # informer 用来维持报文通讯
    informer: Informer
    # 用来记录额外的信息
    # 计算出来的结果将会以key的作为键值，写入信息流。
    __hooks_dict: Dict[str, Callable]

    # 用来停止后台线程
    stopped: Boolean

    def __init__(self, store: Store, federated_world: FederatedWorld, world: World, auto_start: bool = True):
        self.informer = Informer(store, federated_world, world)
        self.stopped = False

        self.__hooks_dict = OrderedDict()

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
        self.__hooks_dict[name] = hook

    def run(self):
        # NOTE：这里使用的是isalive去判断是否杀死monitor
        # 而不是使用world.ALIVE变量
        # world.ALIVE用于确认是否要销毁这个世界
        while self.informer.isalive() and not self.stopped:
            self.informer.set_sys_state()
            self.informer.set_gpu_state()

            for name, hook in self.__hooks_dict.items():
                self.informer.set(name, hook())

            time.sleep(self.informer.world.SLEEPTIME)
        else:
            safe_exited()

    def manual_stop(self):
        """Provide a function to end it manually.
        """
        self.stopped = True
