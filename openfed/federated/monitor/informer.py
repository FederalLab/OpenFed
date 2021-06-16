import datetime
import json
from enum import Enum
from typing import Any, Dict

import openfed
from openfed.utils.types import APPROVED, STATUS

from ..core.federated_c10d import FederatedWorld, Store
from ..world import World
from .gpu_info import getGPUInfo
from .sys_info import getSysInfo

# 以下常量用于设置store里面的键值对。
OPENFED_IDENTIFY = "OPENFED_IDENTIFY"
OPENFED_STATUS = "OPENFED_STATUS"

OPENFED_SYS_INFO = "OPENFED_SYS_INFO"
OPENFED_GPU_INFO = "OPENFED_GPU_INFO"

OPENFED_TASK_INFO = "OPENFED_TASK_INFO"


def to_enum(value, enum_type: Enum):
    for enum in enum_type:
        if enum.value == value:
            return enum
    else:
        raise ValueError(f"{value} is not a valid enum {enum_type}")


def safe_store_set(store: Store, key: str, value: Dict) -> bool:
    # 将数据的解析放在try外面，用来提示更丰富的错误信息。
    jsonstr = json.dumps(value)

    try:
        store.set(key, jsonstr)
        return True
    except Exception as e:
        if openfed.DEBUG:
            # 双方在结束连接时，总会有一方先退出，导致另一方数据读取错误。
            # 这里不是一个bug，所以只是print了异常
            print(e)
        return False


def safe_store_get(store: Store, key: str) -> Dict:
    try:
        jsonbytes = store.get(key)
        # 将数据的解析，放入到try里面。
        # 如果数据解析错误，那么一定是没有读取到完整的数据
        jsonstr = str(jsonbytes, encoding='utf-8')
        info = json.loads(jsonstr)
        return info
    except Exception as e:
        if openfed.DEBUG:
            # 双方在结束连接时，总会有一方先退出，导致另一方数据读取错误。
            # 这里不是一个bug，所以只是print了异常
            print(e)
        return {}


class Informer(object):
    """维护world状态，保证world状态和信息流中的状态是一致的。
    封装kvstore，提供一个更加便捷的接口调用。

    读写规则：读对方的数据，写自己的数据！
    这样可以避免任何形式上的冲突！
    自己的状态不需要读，别人的状态没办法写！
    """
    store: Store
    federated_world: FederatedWorld
    world: World

    def __init__(self, store: Store, federated_world: FederatedWorld, world: World):
        self.federated_world = federated_world
        self.world = world
        self.store = store

        # 写入一个初始信息(为空)
        safe_store_set(self.store, self._i_key, dict())

        self.set_state(STATUS.ZOMBINE)

        # 尝试着读取以下对方的键值，可以用来判断对方是否正常上线。
        # 如果对方没有设置这个值，则会阻塞。
        safe_store_get(self.store, self._u_key)

    @property
    def _i_key(self) -> str:
        """给传入的key加一个自己的后缀
        """
        return OPENFED_IDENTIFY + "_" + ("KING" if self.world.is_king() else "QUEEN")

    @property
    def _u_key(self) -> str:
        """给传入的key加一个对方的后缀
        """
        return OPENFED_IDENTIFY + "_" + ("KING" if not self.world.is_king() else "QUEEN")

    def _write(self, info: Dict) -> bool:
        """Erase old value, write info instead.
        永远都是写到suf_i_key中！
        """
        # 给每一个数据都加入一个时间戳，以保证信息的正确性
        info["timestemp"] = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')

        return safe_store_set(self.store, self._i_key, info)

    def _read(self) -> Dict:
        """永远都是读对方的数据，即_suf_u_key
        """
        info = safe_store_get(self.store, self._u_key)

        if OPENFED_STATUS not in info:
            # 如果没有正确读取到状态的话，那就下线
            info[OPENFED_STATUS] = STATUS.OFFLINE.value

        return info

    def _update(self, info: Dict) -> bool:
        """Update old value with info.
        """
        old_info = self._read()
        old_info.update(info)

        return self._write(old_info)

    def set(self, key: str, value: Any):
        """像字典一样设置键值对。注意：这个值不是直接写在store里，而是写在OPFEN_IFENTITY下面。
        """
        self._update({key: value})

    def get(self, key: str) -> Any:
        # 读取key，如果没有则返回None
        return self._read()[key]

    def alive(self):
        """判断客户端是否在线
        """
        # 首先判断这个world是不是存活的
        return self.world.ALIVE and self.get_state() != STATUS.OFFLINE

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
        return self.get(OPENFED_GPU_INFO)

    def set_gpu_state(self):
        if self.world.APPROVED == APPROVED.ALL or self.world.APPROVED == APPROVED.GPU:
            gpu_state = getGPUInfo()
        else:
            gpu_state = {}
        self.set(OPENFED_GPU_INFO, gpu_state)

    def get_sys_state(self) -> dict:
        return self.get(OPENFED_SYS_INFO)

    def set_sys_state(self):
        if self.world.APPROVED == APPROVED.ALL or self.world.APPROVED == APPROVED.SYS:
            sys_state = getSysInfo()
        else:
            sys_state = {}
        self.set(OPENFED_SYS_INFO, sys_state)
