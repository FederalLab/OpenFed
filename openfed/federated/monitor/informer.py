import json
from typing import Any

import openfed
import openfed.utils as utils
from openfed import CONSTANTS, STATUS
from torch._C._distributed_c10d import Store


class FedDict(dict):
    def jsonize(self) -> str:
        return json.JSONEncoder().encode(self)

    def dictize(self, jsonstr) -> Any:
        self.update(json.JSONDecoder().decode(jsonstr))
        return self


class Informer(object):
    store: Store

    def __init__(self, store: Store):
        self.store = store
        self._write(FedDict())

        # 客户端和服务器端分别写入默认信息
        # 客户端
        if openfed.is_queen():
            # 设置状态
            self.set_state(STATUS.ZOMBINE)  # 表示客户端上线

    def _write(self, feddict: FedDict) -> Any:
        """Erase old value, write feddict instead.
        """
        return self.store.set(CONSTANTS.OPENFED_IDENTIFY.value, feddict.jsonize())

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
        """判断程序是否是存活的（非OFFLINE）
        """
        return self.get_state() == STATUS.OFFLINE

    def get_state(self) -> STATUS:
        state = self.get(CONSTANTS.OPENFED_STATUS.value)
        return utils.to_enum(state, STATUS)

    def set_state(self, state: STATUS):
        self.set(CONSTANTS.OPENFED_STATUS.value, state.value)

    def get_task_info(self) -> dict:
        return self.get(CONSTANTS.OPENFED_TASK_INFO.value)

    def set_task_info(self, task_info: dict):
        self.set(CONSTANTS.OPENFED_TASK_INFO.value, task_info)

    def get_gpu_state(self) -> dict:
        if openfed.is_king():
            # 如果是king，则查看queue的数据
            return self.get(CONSTANTS.OPENFED_QUEUE_GPU_INFO.value)
        else:
            # 如果是queue，则查看king的数据
            return self.get(CONSTANTS.OPENFED_KING_GPU_INFO.value)

    def set_gpu_state(self, gpu_state: dict):
        # 客户端根据自身的身份设置数据
        if openfed.is_king():
            self.set(CONSTANTS.OPENFED_KING_GPU_INFO.value, gpu_state)
        else:
            self.set(CONSTANTS.OPENFED_QUEUE_GPU_INFO.value, gpu_state)

    def get_sys_state(self) -> dict:
        if openfed.is_king():
            # 如果是king，则查看queue的数据
            return self.get(CONSTANTS.OPENFED_QUEUE_SYS_INFO.value)
        else:
            # 如果是queue，则查看king的数据
            return self.get(CONSTANTS.OPENFED_KING_SYS_INFO.value)

    def set_sys_state(self, gpu_state: dict):
        # 客户端根据自身的身份设置数据
        if openfed.is_king():
            self.set(CONSTANTS.OPENFED_KING_SYS_INFO.value, gpu_state)
        else:
            self.set(CONSTANTS.OPENFED_QUEUE_SYS_INFO.value, gpu_state)

