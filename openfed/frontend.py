from typing import Any, Dict, List, Union, overload

from torch import Tensor
from torch.optim import Optimizer

from .federated.federated import Maintainer, Reign, World, default_reign
from .utils.types import FedAddr


class Frontend(object):
    """为客户端提供一个统一简洁的接口！
    对于常规训练，用户只需要接触这个类，就可以解决所有的通讯问题。
    """
    # 一个maintiner用于处理所有连接
    maintiner: Maintainer

    # 一个用于和服务器通信的模块
    reign: Reign

    @overload
    def __init__(self):
        """所有的参数都使用默认的设置。
        其中fed_addr由之后通过build_connection()传入。
        """

    @overload
    def __init__(self,
                 world: World = None,
                 fed_addr: FedAddr = None,
                 fed_addr_file: str = None):
        """指定不同的参数，进行初始化的同时，完成连接。
        记住！这里的fed_addr或者fed_addr_file只能包含一个地址
        且必须包含一个地址，否则报错。
        """

    def __init__(self, **kwargs):
        if len(kwargs) == 0:
            world = World()
            world.set_queen()

            self.world = world
            self.maintiner = None
            self.reign = None
        else:
            world, fed_addr, fed_addr_file = kwargs["world"], kwargs["fed_addr"], kwargs["fed_addr_file"]
            if world is None:
                world = World()
                world.set_queen()
            else:
                assert world.is_queen(), "Frontend must be queen."
            self.maintiner = Maintainer(world, fed_addr, fed_addr_file)

            self.reign = default_reign()

    def build_connection(self, fed_addr: FedAddr):
        self.maintiner = Maintainer(self.world, fed_addr)
        self.reign = default_reign()

    def state_dict_map(self, state_dict: Dict[str, Tensor]):
        self.reign.package.state_dict_map(state_dict)

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
        self.reign.destroy()
