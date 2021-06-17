import json
from collections import OrderedDict
from enum import Enum, unique
from typing import Any, Dict

import openfed
import openfed.utils as utils

from ..core import FederatedWorld, Store, World
from .functional import Collector

# 以下常量用于设置store里面的键值对。
OPENFED_IDENTIFY = "OPENFED_IDENTIFY"
OPENFED_STATUS = "OPENFED_STATUS"

OPENFED_TASK_INFO = "OPENFED_TASK_INFO"

# 所有的操作都是由客户端向服务器端发送请求，服务器端只能应答请求。
# 当服务器完成应答后，会将客户端状态设置成ZOMBINE。
# 如果客户端下线，则程序状态改为OFFINE


@unique
class STATUS(Enum):
    PUSH = "PUSH"  # 把数据推送到服务器
    PULL = "PULL"  # 从服务器拉取数据
    ZOMBINE = "ZOMBINE"  # 当客户端处于其他任何状态时，对于服务器来说，都是ZOMBINE的状态。
    OFFLINE = "OFFLINE"  # 当客户端不在线时，设置成OFFLINE。其余所有状态都表示客户端在线。
    # 因此，客户端程序退出时，应该记得调用相关函数，对状态进行设置。


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
    world: World
    federated_world: FederatedWorld

    _collector_dict: Dict

    def __init__(self):
        self._collector_dict = OrderedDict()

        # 写入一个初始状态，必须采用原始的方式写入，否则程序会因为不同的执行速度而导致读取到无效的信息。
        # 例如：当我们建立连接后，对方会去读取你的键值，如果没有被设置，则会等待，但是一旦被设置了，就会直接读取到结果。
        # 这时候，你要保证你的状态是正确的，否则的话，对方会强制下线
        # 不能通过set_state()来设置！因为set_state会先读取i_key。但是这个时候键值还没有生成
        safe_store_set(self.store, self._i_key, {
                       OPENFED_STATUS: STATUS.ZOMBINE.value})

        # 预先写入task info来防止出错
        self.set_task_info({})

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
        info["timestemp"] = utils.time_string()

        return safe_store_set(self.store, self._i_key, info)

    def _read(self, key: str = None) -> Dict:
        """永远都是读对方的数据，即_suf_u_key
        """
        info = safe_store_get(self.store, self._u_key)

        if OPENFED_STATUS not in info:
            # 如果没有正确读取到状态的话，那就下线
            info[OPENFED_STATUS] = STATUS.ZOMBINE.value
        if key is not None:
            return info[key]
        else:
            return info

    def _update(self, info: Dict) -> bool:
        """Update old value with info.
        """
        # 先把自己的信息读出来，更新一下，在写回去
        # 不要直接调用_read读。read函数默认读取的是对方的信息！
        # 当我们进行更新的时候，是更新自己的信息
        old_info = safe_store_get(self.store, self._i_key)
        old_info.update(info)

        return self._write(old_info)

    def set(self, key: str, value: Any):
        """像字典一样设置键值对。注意：这个值不是直接写在store里，而是写在OPFEN_IFENTITY下面。
        """
        self._update({key: value})

    def get(self, key: str) -> Any:
        # 读取key，如果没有则返回None
        return self._read(key)

    def get_task_info(self) -> dict:
        return self.get(OPENFED_TASK_INFO)

    def set_task_info(self, task_info: dict):
        self.set(OPENFED_TASK_INFO, task_info)

    # 关于状态的一些函数
    def alive(self):
        """判断客户端是否在线
        """
        # 首先判断这个world是不是存活的
        return self.world.ALIVE and self._get_state() != STATUS.OFFLINE

    def _get_state(self) -> STATUS:
        state = self.get(OPENFED_STATUS)
        return to_enum(state, STATUS)

    def _set_state(self, state: STATUS):
        self.set(OPENFED_STATUS, state.value)

    def pulling(self):
        self._set_state(STATUS.PULL)

    def is_pulling(self) -> bool:
        return self._get_state() == STATUS.PULL

    def pushing(self):
        self._set_state(STATUS.PUSH)

    def is_pushing(self) -> bool:
        return self._get_state() == STATUS.PUSH

    def zombine(self):
        self._set_state(STATUS.ZOMBINE)

    def is_zombine(self) -> bool:
        return self._get_state() == STATUS.ZOMBINE

    def offline(self):
        self._set_state(STATUS.OFFLINE)

    def is_offline(self):
        return self.world.ALIVE and self._get_state() == STATUS.OFFLINE

    def register_collector(self, key: str, collector: Collector):
        """请在大多数情况下，使用colector的方式赋值，而不是直接使用set函数。
        set函数不会对键值的冲突进行检测，容易发生错误。
        如果是人物相关的一些信息，比如训练精度等，请使用taskinfo来赋值。
        """
        assert key not in self._collector_dict and key not in [
            OPENFED_STATUS, OPENFED_TASK_INFO]

        self._collector_dict[key] = collector

    def collect(self):
        """运行预设的钩子，去收集相关信息并且上传。

        当你想要用这个来更好的控制你的训练进程的时候，非常有用。
        比如比可以把训练的epoch、learning rate等参数封装成一个collector，
        然后每次训练完直接collect一下，就可以方便的上传到服务器。
        """
        cdict = {}
        for k, f in self._collector_dict.items():
            cdict[k] = f()
        self._update(cdict)
