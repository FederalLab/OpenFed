from typing import Any, Dict, List, Union, overload

from torch import Tensor
from torch.optim import Optimizer

from .common import Address, Peeper, default_address, logger
from .federated.federated import Maintainer, Reign, World, default_reign
import openfed
import warnings


class Frontend(Peeper):
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
                 address: Address):
        """指定不同的参数，进行初始化的同时，完成连接。
        """

    def __init__(self, **kwargs):
        world = kwargs.get('world', None)
        address = kwargs.get('address', None)
        if world is None:
            world = World()
            world.set_queen()
        else:
            world = kwargs['world']
            assert world.is_queen(), "Frontend must be queen."

        if address is None:
            address = default_address

        self.world = world
        self.build_connection(address)

    def build_connection(self, address: Address):
        self.maintainer = Maintainer(self.world, address)
        self.reign = default_reign()

    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        self.reign.set_state_dict(state_dict)

    def pack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.pack_state(obj, keys)

    def unpack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.unpack_state(obj, keys)

    def upload(self) -> bool:
        state = self.reign.upload()
        msg = "Failed to upload data, the server may have shut down."
        if openfed.DEBUG and not state:
            logger.error(msg)
        elif not state:
            logger.warning(msg)
        return state

    def download(self) -> bool:
        state = self.reign.download()
        msg = "Failed to upload data, the server may have shut down."
        if openfed.DEBUG and not state:
            logger.error(msg)
        elif not state:
            logger.warning(msg)
        return state

    def set_task_info(self, task_info: Dict) -> None:
        self.reign.set_task_info(task_info)

    def get_task_info(self) -> Dict:
        return self.reign.get_task_info()

    def set(self, key: str, value: Any) -> None:
        self.reign.set(key, value)

    def get(self, key: str) -> Any:
        return self.reign.get(key)

    def finish(self):
        # 已经完成了训练，退出联邦学习。
        if self.reign is not None:
            self.reign.destroy()

        self.maintainer.manual_stop()

    def __repr__(self):
        return "Frontend"
