# 在这里面定义所有简单的数据类型与全局变量

import json
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, List, TypeVar, Union, overload

from torch import Tensor

A = TypeVar("A", bound='FedAddr')


@dataclass
class FedAddr(object):
    backend: str
    init_method: str = None
    world_size: int = 2
    rank: int = 1
    store = None
    group_name: str = ''

    @classmethod
    def load_from_file(cls, file: str) -> List[A]:
        """从文件加载的方式，创建的fedworld，不支持store的初始化方式！
        """
        with open(file, 'r') as f:
            fed_addr_dict_list = json.load(f)
        fed_addr_list = []
        for fed_addr_dict in fed_addr_dict_list:
            fed_addr_list.append(FedAddr(**fed_addr_dict))
        return fed_addr_list

    @classmethod
    def dump_to_file(cls, file: str, fed_addr_list: List[A]):
        """
            fed_addr_list中的store不会被保存下来。因为不支持从这种方式初始化。
        """
        fed_addr_dict_list = []
        for fed_addr in fed_addr_list:
            fed_addr_dict_list.append(
                dict(backend=fed_addr.backend,
                     init_method=fed_addr.init_method,
                     world_size=fed_addr.world_size,
                     rank=fed_addr.rank,
                     group_name=fed_addr.group_name)
            )
        with open(file, "w") as f:
            json.dump(fed_addr_dict_list, f)


@unique
class APPROVED(Enum):
    GPU = "CPU"
    SYS = "GPU"
    ALL = "ALL"
    NONE = None


@unique
class ROLE(Enum):
    KING = True
    QUEEN = False


# 所有的操作都是由客户端向服务器端发送请求，服务器端只能应答请求。
# 当服务器完成应答后，会将客户端状态设置成ZOMBINE。
# 如果客户端下线，则程序状态改为OFFINE


@unique
class STATUS(Enum):
    PUSH = True  # 把数据推送到服务器
    PULL = False  # 从服务器拉取数据
    ZOMBINE = None  # 当客户端处于其他任何状态时，对于服务器来说，都是ZOMBINE的状态。
    OFFLINE = "OFFLINE"  # 当客户端不在线时，设置成OFFLINE。其余所有状态都表示客户端在线。
    # 因此，客户端程序退出时，应该记得调用相关函数，对状态进行设置。


PACKAGES = TypeVar(
    'PACKAGES', bound=Dict[str, Union[Tensor, Dict[str, Tensor]]])
