from typing import Any, Dict, List, Union, overload

from torch import Tensor
from torch.optim import Optimizer

from .federated.federated import Maintainer, Reign, World, default_reign
from .utils.types import FedAddr, default_fed_addr


class Frontend(object):
    """为客户端提供一个统一简洁的接口！
    对于常规训练，用户只需要接触这个类，就可以解决所有的通讯问题。
    """
    # 一个maintainer用于处理所有连接
    maintainer: Maintainer

    # 一个用于和服务器通信的模块
    reign: Reign

    @overload
    def __init__(self):
        """所有的参数都使用默认的设置。
        """

    @overload
    def __init__(self,
                 world: World,
                 fed_addr: FedAddr):
        """指定不同的参数，进行初始化的同时，完成连接。
        """

    def __init__(self, **kwargs):
        world = kwargs.get('world', None)
        fed_addr = kwargs.get('fed_addr', None)
        if world is None:
            world = World()
            world.set_queen()
        else:
            world = kwargs['world']
            assert world.is_queen(), "Frontend must be queen."

        if fed_addr is None:
            fed_addr = default_fed_addr

        self.world = world
        self.build_connection(fed_addr)

    def build_connection(self, fed_addr: FedAddr):
        self.maintainer = Maintainer(self.world, fed_addr)
        self.reign = default_reign()

    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        self.reign.package.set_state_dict(state_dict)

    def pack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.package.pack_state(obj, keys)

    def unpack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.package.unpack_state(obj, keys)

    def upload(self):
        self.reign.upload()

    def download(self):
        self.reign.download()

    def set_task_info(self, task_info: Dict) -> None:
        self.reign.monitor.set_task_info(task_info)

    def get_task_info(self) -> Dict:
        self.reign.monitor.get_task_info()

    def set(self, key: str, value: Any) -> None:
        self.reign.monitor.set(key, value)

    def get(self, key: str) -> Any:
        return self.reign.monitor.get(key)

    def finish(self):
        # 已经完成了训练，退出联邦学习。
        if self.reign is not None:
            self.reign.destroy()

        self.maintainer.manual_stop()

    def __expr__(self):
        # TODO: 输出一些基本信息
        return "Frontend"
