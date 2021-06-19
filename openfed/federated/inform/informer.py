import json
from enum import Enum, unique
from typing import Any, Dict

import openfed
import openfed.utils as utils
from openfed.common import Hook, logger

from ...utils import openfed_class_fmt
from ..core import FederatedWorld, Store, World
from .functional import Collector, SystemInfo

OPENFED_IDENTIFY = "OPENFED_IDENTIFY"
OPENFED_STATUS = "OPENFED_STATUS"

OPENFED_TASK_INFO = "OPENFED_TASK_INFO"


@unique
class STATUS(Enum):
    PUSH = "PUSH"  # push data to the other end.
    PULL = "PULL"  # pull data from the other end.
    ZOMBINE = "ZOMBINE"  # when there is no request.
    OFFLINE = "OFFLINE"  # offline.


def to_enum(value, enum_type: Enum):
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
        if openfed.DEBUG.is_debug:
            logger.warning(e)
        return False


def safe_store_get(store: Store, key: str) -> Dict:
    try:
        jsonbytes = store.get(key)
        jsonstr = str(jsonbytes, encoding='utf-8')
        info = json.loads(jsonstr)
        return info
    except Exception as e:
        if openfed.DEBUG.is_debug:
            logger.warning(e)
        return {}


class Informer(Hook):
    """Informer: keep the real time communication between each other via string.
    NOTE: READ the other side information, WRITE self side information.
    """
    store: Store
    world: World
    federated_world: FederatedWorld

    def __init__(self):
        # write self._i_key to initialize the key value store.
        safe_store_set(self.store, self._i_key, {
                       OPENFED_STATUS: STATUS.ZOMBINE.value})

        # pre-write task_info keys.
        self.set_task_info({})

        # register a default collector
        self.register_collector(SystemInfo.bounding_name, SystemInfo())

        self.scatter()

        # Fetch data at last
        # try to read _u_key from the other end to make sure it is online.
        safe_store_get(self.store, self._u_key)

        self.collect()

    @property
    def _i_key(self) -> str:
        return OPENFED_IDENTIFY + "_" + ("KING" if self.world.king else "QUEEN")

    @property
    def _u_key(self) -> str:
        return OPENFED_IDENTIFY + "_" + ("KING" if not self.world.king else "QUEEN")

    def _write(self, info: Dict) -> bool:
        """Write info to self._i_key.
        """
        info["timestemp"] = utils.time_string()
        return safe_store_set(self.store, self._i_key, info)

    def _read(self, key: str = None) -> Dict:
        """Read message from self._u_key.
        """
        info = safe_store_get(self.store, self._u_key)

        if OPENFED_STATUS not in info:
            # set as ZOMBINE, not OFFLINE.
            info[OPENFED_STATUS] = STATUS.ZOMBINE.value
        if key is not None:
            return info[key]
        else:
            return info

    def _update(self, info: Dict) -> bool:
        """rewrite the old message in kv-store.
        """
        # read i_key information, then update it
        old_info = safe_store_get(self.store, self._i_key)
        old_info.update(info)
        return self._write(old_info)

    def set(self, key: str, value: Any):
        self._update({key: value})

    def get(self, key: str) -> Any:
        return self._read(key)

    @property
    def task_info(self) -> Dict:
        return self.get(OPENFED_TASK_INFO)

    def set_task_info(self, task_info: Dict):
        self.set(OPENFED_TASK_INFO, task_info)

    def _get_state(self) -> STATUS:
        state = self.get(OPENFED_STATUS)
        return to_enum(state, STATUS)

    def _set_state(self, state: STATUS):
        self.set(OPENFED_STATUS, state.value)

    @property
    def alive(self):
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

    def zombine(self):
        """Set state to zombine
        """
        self._set_state(STATUS.ZOMBINE)

    @property
    def is_zombine(self) -> bool:
        return self._get_state() == STATUS.ZOMBINE

    def offline(self):
        """Set state to offline.
        """
        self._set_state(STATUS.OFFLINE)

    @property
    def is_offline(self):
        return self.world.ALIVE and self._get_state() == STATUS.OFFLINE

    def register_collector(self, key: str, collector: Collector):
        """Use collector to add new information is strongely recommanded.
        """
        assert key not in [OPENFED_STATUS, OPENFED_TASK_INFO]

        super().register_hook(key=key, func=collector)

    def collect(self):
        """Collect message from other side.
        """
        for k, f in self.hook_dict.items():
            f.load_message(self.get(k))
            if openfed.VERBOSE.is_verbose:
                logger.info(f)

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
