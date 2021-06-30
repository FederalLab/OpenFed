import json
from enum import Enum, unique
from typing import Any, Callable, Dict

import openfed.utils as utils
from openfed.common import Hook, logger
from openfed.utils import openfed_class_fmt
from random_words import RandomWords

from ..space import Country, Store, World
from ..utils import auto_filterout, auto_offline
from ..utils.exception import (BuildReignFailed, InvalidStoreReading,
                               InvalidStoreWriting)
from .functional import Collector, GPUInfo, Register, SystemInfo

rw = RandomWords()


OPENFED_IDENTIFY = "OPENFED_IDENTIFY"
OPENFED_STATUS = "OPENFED_STATUS"

OPENFED_TASK_INFO = "OPENFED_TASK_INFO"
NICK_NAME = "NICK_NAME"


@unique
class STATUS(Enum):
    PUSH = "PUSH"  # push data to the other end.
    PULL = "PULL"  # pull data from the other end.
    ZOMBIE = "ZOMBIE"  # when there is no request.
    OFFLINE = "OFFLINE"  # offline.


def to_enum(value, enum_type: Enum) -> Enum:
    for enum in enum_type:
        if enum.value == value:
            return enum
    else:
        raise ValueError(f"{value} is not a valid enum {enum_type}.")


def safe_store_set(store: Store, key: str, value: Dict) -> bool:
    jsonstr = json.dumps(value)

    try:
        store.set(key, jsonstr)
        return True
    except Exception as e:
        raise InvalidStoreWriting(e)


def safe_store_get(store: Store, key: str) -> Dict:
    try:
        jsonbytes = store.get(key)
        jsonstr = str(jsonbytes, encoding='utf-8')
        return json.loads(jsonstr)
    except Exception as e:
        raise InvalidStoreReading(e)


class Informer(Hook):
    """Informer: keep the real time communication between each other via string.
    NOTE: READ the other side information, WRITE self side information.
    """
    store: Store
    world: World
    country: Country

    # Sometimes, the read operation will be failed for unknown reasons.
    # Essentially when we do simulation experiments on a single node.
    # In order to avoid this error, we always backup the last info
    # if failed, return this instead.
    _backup_info: Dict[str, Any]

    # you should ignore this dict. it will make you feel confused.
    _do_not_access_backup_info: Dict[str, Any]

    # indicates whether current data is
    fresh_read: bool

    # do not set this manually!
    _nick_name: str = None

    def __init__(self):
        self._do_not_access_backup_info = {}
        # write self._i_key to initialize the key value store.
        safe_store_set(self.store, self._i_key, {
                       OPENFED_STATUS: STATUS.ZOMBIE.value})

        # set nick name if leader
        if self.world.leader:
            safe_store_set(self.store, NICK_NAME, rw.random_word())

        # pre-write task_info keys.
        self.set_task_info({})

        # register a default collector
        self.register_collector(SystemInfo())
        self.register_collector(GPUInfo())

        self.scatter()

        # Fetch data at last
        # try to read _u_key from the other end to make sure it is online.
        try:
            self._backup_info = safe_store_get(self.store, self._u_key)
        except InvalidStoreReading as e:
            raise BuildReignFailed(e)

        # Run at the initialize state.
        self.collect()
        self.fresh_read = True
        # make a copy
        self._nick_name = self.nick_name

    @property
    @auto_offline
    def nick_name(self) -> str:
        if self._nick_name:
            return self._nick_name
        else:
            return safe_store_get(self.store, NICK_NAME)

    @property
    def _i_key(self) -> str:
        return OPENFED_IDENTIFY + "_" + ("LEADER" if self.world.leader else "FOLLOWER")

    @property
    def _u_key(self) -> str:
        return OPENFED_IDENTIFY + "_" + ("LEADER" if not self.world.leader else "FOLLOWER")

    @auto_filterout
    def _write(self, info: Dict[str, str]) -> bool:
        """Write info to self._i_key.
        """
        info["timestemp"] = utils.time_string()
        return safe_store_set(self.store, self._i_key, info)

    def _update(self, info: Dict[str, str]) -> bool:
        """rewrite the old message in kv-store.
        """
        # read i_key information, then update it
        try:
            old_info = safe_store_get(self.store, self._i_key)
        except InvalidStoreReading as e:
            logger.debug(e)
            old_info = self._do_not_access_backup_info
        except Exception as e:
            old_info = self._do_not_access_backup_info
        finally:
            old_info.update(info)
            self._do_not_access_backup_info = old_info

            return self._write(old_info)

    def _read(self, key: str = None) -> Dict:
        """Read message from self._u_key.
        """
        try:
            info = safe_store_get(self.store, self._u_key)
            self.fresh_read = True
        except InvalidStoreReading as e:
            logger.debug(e)
            info = self._backup_info
            # use the cached one instead.
            # but at the same time, we need to set the state as zombie
            # otherwise the last state value may make the progress get stuck.
            # The server is quiet stable, if read failed, we think it is offline.
            # But client sometimes may be unstable, if read failed, we will assume it
            # go into offline.
            info[OPENFED_STATUS] = STATUS.OFFLINE.value if self.world.follower else STATUS.ZOMBIE.value
            self.fresh_read = False
        finally:
            self._backup_info = info
            return info[key] if key else info

    def _must_fresh_read(func: Callable):
        """A decorate function that will raise error if the data is refresh.
        """

        def wrapper(self, *args, **kwargs):
            output = func(self, *args, **kwargs)
            if not self.fresh_read:
                logger.debug(
                    "Use an cached value instead a fresh required data."
                    "Which may cause Error."
                    f"func: {func}"
                    f"args: {args}"
                    f"kwargs: {kwargs}")
            return output
        return wrapper

    def set(self, key: str, value: Any):
        self._update({key: value})

    def get(self, key: str) -> Any:
        return self._read(key)

    @property
    @_must_fresh_read
    def task_info(self) -> Dict[str, Any]:
        return self.get(OPENFED_TASK_INFO)

    def set_task_info(self, task_info: Dict[str, Any]):
        self.set(OPENFED_TASK_INFO, task_info)

    def _get_state(self) -> STATUS:
        state = self.get(OPENFED_STATUS)
        return to_enum(state, STATUS)

    def _set_state(self, state: STATUS):
        self.set(OPENFED_STATUS, state.value)

    @property
    def alive(self) -> bool:
        """opposite to self.offline
        """
        return self.world.ALIVE and self._get_state() != STATUS.OFFLINE

    def pulling(self):
        """Set state to pull.
        """
        self._set_state(STATUS.PULL)

    @property
    def is_pulling(self) -> bool:
        return self._get_state() == STATUS.PULL

    def pushing(self):
        """Set state to push.
        """
        self._set_state(STATUS.PUSH)

    @property
    def is_pushing(self) -> bool:
        return self._get_state() == STATUS.PUSH

    def zombie(self):
        """Set state to zombie
        """
        self._set_state(STATUS.ZOMBIE)

    @property
    def is_zombie(self) -> bool:
        return self._get_state() == STATUS.ZOMBIE

    def offline(self):
        """Set state to offline.
        """
        self._set_state(STATUS.OFFLINE)

    @property
    def is_offline(self) -> bool:
        return self.world.ALIVE and self._get_state() == STATUS.OFFLINE

    @Register.add_to_pool
    def register_collector(self, collector: Collector):
        """Use collector to add new information is strongely recommended.
        """
        super().register_hook(key=collector.bounding_name, func=collector)

    @auto_filterout
    def collect(self):
        """Collect message from other side.
        """
        # read all collection information
        # key = None will return the whole info dictionary.
        info = self.get(key=None)
        for key, value in info.items():
            if key.startswith("Collector"):
                # don't forget () operation.
                obj = Register(key, self)()
                if obj is not None:
                    obj.load_message(value)
                    logger.debug(obj)

    def scatter(self):
        """Scatter self.hook information to the other end.
        """
        cdict = {}
        for k, f in self.hook_dict.items():
            cdict[k] = f()
        self._update(cdict)

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Informer",
            description=str(list(self.hook_dict.keys())),
        )
