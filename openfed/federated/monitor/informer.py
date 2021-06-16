import datetime
import json
from enum import Enum
from typing import Any, Dict, final

from openfed.utils.types import APPROVED, STATUS

from ..core.federated_c10d import FederatedWorld, Store
from ..world import World
from .gpu_info import getGPUInfo
from .sys_info import getSysInfo
import openfed

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
        return json.dumps(self)

    def dictize(self, jsonstr: bytes) -> Any:
        jsonstr = str(jsonstr, encoding="utf-8")
        self.update(json.loads(jsonstr))
        return self


def safe_store_set(store: Store, key: str, value: str) -> bool:
    try:
        store.set(key, value)
        return True
    except Exception as e:
        if openfed.DEBUG:
            # 双方在结束连接时，总会有一方先退出，导致另一方数据读取错误。
            # 这里不是一个bug，所以只是print了异常
            print(e)
        return False


def safe_store_get(store: Store, key: str) -> str:
    try:
        return store.get(key)
    except Exception as e:
        if openfed.DEBUG:
            # 双方在结束连接时，总会有一方先退出，导致另一方数据读取错误。
            # 这里不是一个bug，所以只是print了异常
            print(e)
        return ""


class Informer(object):
    """维护world状态，保证world状态和信息流中的状态是一致的。
    封装kvstore，提供一个更加便捷的接口调用。

    TODO:注意处理客户端和服务器端进行写入操作时，对键值的冲突。
    """
    store: Store
    federated_world: FederatedWorld
    world: World

    # 这个变量的作用主要是为了防止程序以外结束时，无法读取任何信息。
    # 这里可以保证读取的是之前的内容
    _buf_dict: Dict

    def __init__(self, store: Store, federated_world: FederatedWorld, world: World):
        self.federated_world = federated_world
        self.world = world
        self.store = store

        # 开始设置其他东西之前，先写入这个key，防止后面出现无数据可取！
        if not safe_store_set(self.store, OPENFED_IDENTIFY, "{}"):
            raise RuntimeError("Initialize store failed")

        if self.world.is_king():
            safe_store_set(self.store, OPENFED_IDENTIFY+"_KING", "REGISTERED")
            safe_store_get(self.store, OPENFED_IDENTIFY+"_QUEEN")
        else:
            self.set_state(STATUS.ZOMBINE)  # 表示客户端上线
            safe_store_set(self.store, OPENFED_IDENTIFY+"_QUEEN", "REGISTERED")
            safe_store_get(self.store, OPENFED_IDENTIFY+"_KING")

        # 保证读的数据是一致的在最开始的时候
        self._buf_dict = self._read()

    def _write(self, feddict: FedDict) -> bool:
        """Erase old value, write feddict instead.
        """
        # 给每一个数据都加入一个时间戳，以保证信息的正确性
        feddict["timestemp"] = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')

        return safe_store_set(self.store, OPENFED_IDENTIFY, feddict.jsonize())

    def _read(self) -> Dict:
        fed_dict = FedDict()
        raw_str = safe_store_get(self.store, OPENFED_IDENTIFY)

        try:
            fed_dict.dictize(raw_str)
            self._buf_dict = fed_dict
        except Exception as e:
            if openfed.DEBUG:
                # 这里不一定是bug。但是进行字典化出问题了一般是没有读取到正确的数据格式。
                print(e)
            # 说明对方已下线（不知原因的下线。）
            # 把我方读取到的状态也下线
            self._buf_dict[OPENFED_STATUS] = STATUS.OFFLINE.value
            fed_dict = self._buf_dict
        finally:
            if OPENFED_STATUS not in fed_dict:
                # 如果没有正确读取到状态的话，那就下线
                fed_dict[OPENFED_STATUS] = STATUS.OFFLINE.value

        return fed_dict

    def _update(self, feddict: FedDict) -> Any:
        """Update old value with feddict.
        """
        old_feddict = self._read()
        old_feddict.update(feddict)
        return self._write(old_feddict)

    def set(self, key: str, value: Any):
        """像字典一样设置键值对。注意：这个值不是直接写在store里，而是写在OPFEN_IFENTITY下面。
        """
        feddict = FedDict()
        feddict[key] = value
        self._update(feddict)

    def get(self, key: str) -> Any:
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
