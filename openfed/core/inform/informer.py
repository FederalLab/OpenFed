# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import json
from enum import Enum, unique
from typing import Any, Callable, Dict

import openfed.utils as utils
from openfed.common import Hook, TaskInfo, logger
from openfed.utils import openfed_class_fmt
from random_words import RandomWords

from ..space import Country, Store, World
from ..utils.exceptions import (BuildReignFailed, InvalidStoreReading,
                               InvalidStoreWriting)
from .collector import Collector, GPUInfo, Register, SystemInfo

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


push = STATUS.PUSH.value
pull = STATUS.PULL.value
zombie = STATUS.ZOMBIE.value
offline = STATUS.OFFLINE.value


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

        # set nick name first!
        if self.world.leader:
            safe_store_set(self.store, NICK_NAME, rw.random_word())

        # write self._i_key to initialize the key value store.
        safe_store_set(self.store, self._i_key, {
                       OPENFED_STATUS: zombie, OPENFED_TASK_INFO: TaskInfo().as_dict()})

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

    def _write(self, info: Dict[str, str]) -> bool:
        """Write info to self._i_key.
        """
        info["timestemp"] = utils.time_string()
        try:
            flag = safe_store_set(self.store, self._i_key, info)
        except InvalidStoreWriting as e:
            logger.info("Write Failed.")
            flag = False
        finally:
            return flag

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
            info[OPENFED_STATUS] = offline if self.world.follower else zombie
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
    def task_info(self) -> TaskInfo:
        return TaskInfo().load_dict(self.get(OPENFED_TASK_INFO))

    def set_task_info(self, task_info: TaskInfo):
        self.set(OPENFED_TASK_INFO, task_info.as_dict())

    def _get_state(self) -> str:
        return self.get(OPENFED_STATUS)

    def _set_state(self, state: str):
        self.set(OPENFED_STATUS, state)

    @property
    def alive(self) -> bool:
        """opposite to self.offline
        """
        return self.world.ALIVE and self._get_state() != offline

    def pulling(self):
        """Set state to pull.
        """
        self._set_state(pull)

    @property
    def is_pulling(self) -> bool:
        return self._get_state() == pull

    def pushing(self):
        """Set state to push.
        """
        self._set_state(push)

    @property
    def is_pushing(self) -> bool:
        return self._get_state() == push

    def zombie(self):
        """Set state to zombie
        """
        self._set_state(zombie)

    @property
    def is_zombie(self) -> bool:
        return self._get_state() == zombie

    def offline(self):
        """Set state to offline.
        """
        self._set_state(offline)

    @property
    def is_offline(self) -> bool:
        return self.world.ALIVE and self._get_state() == offline

    @Register.add_to_pool
    def register_collector(self, collector: Collector):
        """Use collector to add new information is strongely recommended.
        """
        super().register_hook(key=collector.bounding_name, func=collector)

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
                if obj is not None and self.world.leader and obj.leader_collector or \
                        self.world.follower and obj.follower_collector:
                    obj.load_message(value)
                    logger.debug(obj)

    def scatter(self):
        """Scatter self.hook information to the other end.
        """
        cdict = {}
        for k, f in self.hook_dict.items():
            if self.world.leader and f.leader_scatter or\
                    self.world.follower and f.follower_scatter:
                cdict[k] = f()
        self._update(cdict)

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Informer",
            description=str(list(self.hook_dict.keys())),
        )
