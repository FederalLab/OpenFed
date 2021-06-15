import datetime
import json
from enum import Enum
from typing import Any

from openfed.utils.types import APPROVED, STATUS

from ..core.federated_c10d import FederatedWorld, Store
from ..world import World
from .gpu_info import getGPUInfo
from .sys_info import getSysInfo

# 以下常量用于设置store里面的键值对。
OPENFED_IDENTIFY = "OPENFED_IDENTIFY"
OPENFED_STATUS = "OPENFED_STATUS"
OPENFED_KING_SYS_INFO = "OPENFED_KING_SYS_INFO"
OPENFED_KING_GPU_INFO = "OPENFED_KING_GPU_INFO"
OPENFED_QUEUE_SYS_INFO = "OPENFED_QUEUE_SYS_INFO"
OPENFED_QUEUE_GPU_INFO = "OPENFED_QUEUE_GPU_INFO"

OPENFED_TASK_INFO = "OPENFED_TASK_INFO"


def to_enum(value, enum_type: Enum):
    for enum in enum_type:
        if enum.value == value:
            return enum
    else:
        raise ValueError(f"{value} is not a valid enum {enum_type}")


class FedDict(dict):
    def jsonize(self) -> str:
        return json.JSONEncoder().encode(self)

    def dictize(self, jsonstr) -> Any:
        self.update(json.JSONDecoder().decode(jsonstr))
        return self


class Informer(object):
    """维护world状态，保证world状态和信息流中的状态是一致的。
    封装kvstore，提供一个更加便捷的接口调用。
    """
    store: Store
    federated_world: FederatedWorld
    world: World

    def __init__(self, store: Store, federated_world: FederatedWorld, world: World):
        self.federated_world = federated_world
        self.world = world
        self.store = store
        self._write(FedDict())

        # 客户端和服务器端分别写入默认信息
        if self.world.is_queen():
            # 设置状态
            self.set_state(STATUS.ZOMBINE)  # 表示客户端上线

    def _write(self, feddict: FedDict) -> Any:
        """Erase old value, write feddict instead.
        """
        # 给每一个数据都加入一个时间戳，以保证信息的正确性
        feddict["timestemp"] = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')
        return self.store.set(OPENFED_IDENTIFY, feddict.jsonize())

    def _update(self, feddict: FedDict) -> Any:
        """Update old value with feddict.
        """
        return self._write(self.read()._update(feddict))

    def set(self, key: str, value: Any):
        """像字典一样设置键值对。注意：这个值不是直接写在store里，而是写在OPFEN_IFENTITY下面。
        """
        feddict = FedDict()
        feddict.set(key, value)
        self._update(feddict)

    def get(self, key: str) -> Any:
        return self.read()[key]

    def __del__(self):
        # 调用析构函数，保证客户端退出时，服务器端察觉相应的状态信息，并且销毁连接。
        self.set_state(STATUS.OFFLINE)
        super().__del__()

    def alive(self):
        """判断客户端是否在线
        """
        return self.get_state() != STATUS.OFFLINE

    def get_state(self) -> STATUS:
        state = self.get(OPENFED_STATUS)
        return to_enum(state, STATUS)

    def set_state(self, state: STATUS):
        self.set(OPENFED_STATUS, state.value)

    def get_task_info(self) -> dict:
        return self.get(OPENFED_TASK_INFO)

    def set_task_info(self, task_info: dict):
        self.set(OPENFED_TASK_INFO, task_info)

    def get_gpu_state(self) -> dict:
        if self.world.is_king():
            # 如果是king，则查看queue的数据
            return self.get(OPENFED_QUEUE_GPU_INFO)
        else:
            # 如果是queue，则查看king的数据
            return self.get(OPENFED_KING_GPU_INFO)

    def set_gpu_state(self):
        if self.world.APPROVED == APPROVED.ALL or self.world.APPROVED == APPROVED.GPU:
            gpu_state = getGPUInfo()
        else:
            gpu_state = {}
        # 客户端根据自身的身份设置数据
        if self.world.is_king():
            self.set(OPENFED_KING_GPU_INFO, gpu_state)
        else:
            self.set(OPENFED_QUEUE_GPU_INFO, gpu_state)

    def get_sys_state(self) -> dict:
        if self.world.is_king():
            # 如果是king，则查看queue的数据
            return self.get(OPENFED_QUEUE_SYS_INFO)
        else:
            # 如果是queue，则查看king的数据
            return self.get(OPENFED_KING_SYS_INFO)

    def set_sys_state(self):
        if self.world.APPROVED == APPROVED.ALL or self.world.APPROVED == APPROVED.SYS:
            sys_state = getSysInfo()
        else:
            sys_state = {}
        # 客户端根据自身的身份设置数据
        if self.world.is_king():
            self.set(OPENFED_KING_SYS_INFO, sys_state)
        else:
            self.set(OPENFED_QUEUE_SYS_INFO, sys_state)
