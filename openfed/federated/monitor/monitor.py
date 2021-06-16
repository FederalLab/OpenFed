import time
from collections import OrderedDict
from threading import Thread
from typing import Callable, Dict

from ..core.federated_c10d import FederatedWorld, Store
from ..utils.safe_exited import safe_exited
from ..world import World
from .informer import Informer


class Monitor(Informer, Thread):
    """自动启动一个新的线程进行状态的监督。
    这个线程的作用是把当前的系统状态，定期在服务器和客户端之间进行同步。
    """
    # 用来记录额外的信息
    # 计算出来的结果将会以key的作为键值，写入信息流。
    __hooks_dict: Dict[str, Callable]

    # 用来停止后台线程
    stopped: bool

    def __init__(self, store: Store, federated_world: FederatedWorld, world: World, auto_start: bool = True):
        super().__init__(store, federated_world, world)
        super(Informer, self).__init__()
        self.stopped = False

        self.__hooks_dict = OrderedDict()

        if auto_start:
            self.start()

    def register_hook(self, name: str, hook: Callable, auto_prefix: bool = True):
        """

        Args: 
            auto_prefix：如果为True，会根据身份自动添加一个后缀到name里面。

        hook计算出来的数据，将会以name为键值，向外传输。
        hook应该是闭包函数，调用过程不会传入任何参数。

        .. node::
            这里的name，不能和任何现有的系统预置的键值重复！
            这里注册的hook，必须返回一个字典，后者可以被josnize的对象！
        """
        if auto_prefix:
            name = f'{name}_{"KING" if self.world.is_king() else "QUEEN"}'
        self.__hooks_dict[name] = hook

    def run(self):
        # NOTE：这里使用的是isalive去判断是否杀死monitor
        # 而不是使用world.ALIVE变量
        # world.ALIVE用于确认是否要销毁这个世界
        while self.is_alive() and not self.stopped:
            self.set_sys_state()
            self.set_gpu_state()

            for name, hook in self.__hooks_dict.items():
                self.set(name, hook())

            time.sleep(self.world.SLEEPTIME)
        else:
            safe_exited()

    def manual_stop(self):
        """Provide a function to end it manually.
        """
        self.stopped = True
