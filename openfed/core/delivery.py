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
import time
from collections import defaultdict
from datetime import timedelta
from enum import Enum, unique
from typing import Any, Callable, Dict, Tuple, Union

import openfed
import openfed.utils as utils
from bidict import bidict
from openfed.common import ASYNC_OP, Hook, Package, TaskInfo, logger
from openfed.utils import openfed_class_fmt, tablist
from random_words import RandomWords
from torch import Tensor
from torch._C._distributed_c10d import Work

from .collector import Collector, GPUInfo, Register, SystemInfo
from .cypher import Cypher, FormatCheck
from .federated.functional import gather_object
from .space import Country, ProcessGroup, Store, World
from .utils.exceptions import (BuilddeliveryFailed, ConnectTimeout,
                               DeviceOffline, InvalidStoreReading,
                               InvalidStoreWriting, WrongState)
from .utils.register import register

rw = RandomWords()


OPENFED_IDENTIFY = "OPENFED_IDENTIFY"
OPENFED_STATUS   = "OPENFED_STATUS"

OPENFED_TASK_INFO = "OPENFED_TASK_INFO"
NICK_NAME         = "NICK_NAME"


@unique
class STATUS(Enum):
    PUSH    = "PUSH"  # push data to the other end.
    PULL    = "PULL"  # pull data from the other end.
    ZOMBIE  = "ZOMBIE"  # when there is no request.
    OFFLINE = "OFFLINE"  # offline.


push    = STATUS.PUSH.value
pull    = STATUS.PULL.value
zombie  = STATUS.ZOMBIE.value
offline = STATUS.OFFLINE.value


def safe_store_set(store: Store, key: str, value: Dict) -> bool:
    jsonstr = json.dumps(value)

    try:
        store.set(key, jsonstr)
        return True
    except Exception as e:
        raise InvalidStoreWriting(e)


def safe_store_get(store: Store, key: str) -> Dict[str, Any]:
    try:
        jsonbytes = store.get(key)
        jsonstr = str(jsonbytes, encoding='utf-8')
        return json.loads(jsonstr)
    except Exception as e:
        raise InvalidStoreReading(e)


def _must_fresh_read(func):
    """A decorate function that will raise error if the data is refresh.
    """

    def must_fresh_read(self, *args, **kwargs):
        output = func(self, *args, **kwargs)
        if not self.fresh_read:
            logger.debug(
                "Use an cached value instead a fresh required data."
                "Which may cause Error."
                f"func: {func}"
                f"args: {args}"
                f"kwargs: {kwargs}")
        return output
    return must_fresh_read


class Delivery(Hook, Package):
    """Contains all communication functions in Delivery.
    """
    store  : Store
    pg     : ProcessGroup
    world  : World
    country: Country

    # handler, step function, timestamp
    _download_hang_up: Tuple[Work, Callable, int]
    _upload_hang_up  : Tuple[Work, Callable, int]

    download_hang_up: bool
    upload_hang_up  : bool

    # Sometimes, the read operation will be failed for unknown reasons.
    # Essentially when we do simulation experiments on a single node.
    # In order to avoid this error, we always backup the last info
    # if failed, return this instead.
    _backup_info: Dict[str, Any]

    # you should ignore this dict. it will make you feel confused.
    _i_backup_info: Dict[str, Any]

    # indicates whether current data is
    fresh_read: bool

    # do not set this manually!
    _nick_name       : Union[None, str]
    key_tensor_bidict: bidict
    packages         : Dict[str, Dict[str, Tensor]]

    leader_rank  : int = 0
    follower_rank: int = 1

    def __init__(self,
                 store  : Store,
                 pg     : ProcessGroup,
                 country: Country,
                 world  : World,
                 ) -> None:
        self.pg             = pg
        self.store          = store
        self.country        = country
        self.world          = world
        self._i_backup_info = {}

        # set nick name first!
        if self.world.leader:
            safe_store_set(self.store, NICK_NAME, rw.random_word())

        # write self._i_key to initialize the key value store.
        safe_store_set(self.store, self._i_key, {
                       OPENFED_STATUS: zombie, OPENFED_TASK_INFO: TaskInfo().info_dict})

        # register a default collector
        self.register_collector(SystemInfo())
        self.register_collector(GPUInfo())

        self.scatter()

        # Fetch data at last
        # try to read _u_key from the other end to make sure it is online.
        try:
            self._backup_info = safe_store_get(self.store, self._u_key)
        except InvalidStoreReading as e:
            raise BuilddeliveryFailed(e)

        # Run at the initialize state.
        self.collect()
        self.fresh_read = True
        # make a copy
        self._nick_name        = None
        self._nick_name        = self.nick_name
        self.key_tensor_bidict = bidict()
        self.packages          = defaultdict(dict)
        self.register_cypher(FormatCheck())

        self.download_hang_up: bool = False
        self.upload_hang_up  : bool = False

    @property
    def upload_version(self) -> int:
        try:
            return self.get("upload_version")
        except Exception as e:
            return -1

    @property
    def download_version(self) -> int:
        try:
            return self.get("download_version")
        except Exception as e:
            return -1

    def transfer(self,
                 to       : bool,
                 handler  : Any = None,
                 tic      : float = None,
                 step_func: Callable = None) -> bool: 
        """
        Args:
            to: it true, transfer data to other side.
            handler: if not None, call handler.wait().
            tic: start counting time.
        """
        if self.is_offline:
            raise DeviceOffline(self)

        def _state():
            return self.is_pulling if to else self.is_pushing

        # logic judge
        if self.world.follower:
            # set state first
            if to:
                self.pushing()
            else:
                self.pulling()
            # wait until satisfied
            tic = time.time() if not tic else tic

            while not _state():
                if self.is_offline:
                    raise DeviceOffline(self)
                toc = time.time()
                if timedelta(seconds=toc-tic) > openfed.DEFAULT_PG_TIMEOUT:
                    raise ConnectTimeout(self)
                time.sleep(0.1)
        else:
            # check state first
            if not _state():
                raise WrongState(self)
            # set state
            if to:
                self.pushing()
            else:
                self.pulling()
        # transfer
        if handler:
            handler.wait()
            if step_func is not None:
                step_func()
        else:
            if to:
                self.push()
            else:
                self.pull()
        self.zombie()
        return True

    def deal_with_hang_up(self) -> bool:
        """Dealing with the handler for hang up operations.
        """
        if self.upload_hang_up:
            handler, step_func, tic = self._upload_hang_up
            to = True
        elif self.download_hang_up:
            handler, step_func, tic = self._download_hang_up
            to = False
        else:
            return False

        try:
            flag = self.transfer(to=to, handler=handler,
                                 tic=tic, step_func=step_func)
        except WrongState as e:
            return False

        if handler.is_completed():
            if self.upload_hang_up:
                del self._upload_hang_up
                self.upload_hang_up = False
            if self.download_hang_up:
                del self._download_hang_up
                self.download_hang_up = False
            return flag
        else:
            toc = time.time()
            if timedelta(seconds=toc-tic) > openfed.DEFAULT_PG_TIMEOUT:
                raise ConnectTimeout(f"Timeout while waiting {self}")
            else:
                # keep waiting
                return False

    def upload(self, version: int) -> bool:
        """Upload packages date to the other end.
        """

        # set version on task info
        self.set("upload_version", version)

        if ASYNC_OP.is_async_op:
            handle, step_func = self.push()
            # store the necessary message, and hang up begining time.
            self._upload_hang_up = (handle, step_func, time.time()) # type: ignore
            self.upload_hang_up  = True
            return False
        else:
            return self.transfer(to=True)

    def download(self, version: int) -> bool:
        """Download packages from other end.
        """

        # set version
        self.set("download_version", version)

        if ASYNC_OP.is_async_op:
            handle, step_func = self.pull()
            self._download_hang_up = (handle, step_func, time.time()) # type: ignore
            self.download_hang_up = True
            return False
        else:
            return self.transfer(to=False)

    @classmethod
    def delivery_generator(cls):
        """Return a generator to iterate over all deliverys.
        """
        for _, world in register:
            if world is None and not world.ALIVE:
                continue
            for pg, delivery in world:
                if delivery is None and not world.ALIVE:
                    continue
                yield delivery
                world._current_pg = pg

    @classmethod
    def default_delivery(cls):
        """Return the only delivery. If more then one, raise warning.
        """
        if len(register) == 0:
            logger.debug("Empty world.")
        if len(register) > 1:
            logger.debug("More than one register world, use the earliest one.")
        return register.default_world.default_delivery

    @property
    def nick_name(self) -> Any:
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
            return safe_store_set(self.store, self._i_key, info)
        except InvalidStoreWriting as e:
            logger.info("Write Failed.")
            return False

    def _update(self, info: Dict[str, str]) -> bool:
        """rewrite the old message in kv-store.
        """
        old_info = self._i_backup_info
        # read i_key information, then update it
        try:
            old_info = safe_store_get(self.store, self._i_key)
        except InvalidStoreReading as e:
            logger.debug(e)
            ...
        except Exception as e:
            raise e
        finally:
            old_info.update(info)
            self._i_backup_info = old_info

            return self._write(old_info)

    def _read(self, key: str = None) -> Dict:
        """Read message from self._u_key.
        """
        info = self._backup_info
        try:
            info = safe_store_get(self.store, self._u_key)
            self.fresh_read = True
        except InvalidStoreReading as e:
            logger.debug(e)
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

    def set(self, key: str, value: Any):
        self._update({key: value})

    def get(self, key: Union[None, str]) -> Any:
        return self._read(key)

    @property
    @_must_fresh_read
    def task_info(self) -> TaskInfo:
        return TaskInfo().load_dict(self.get(OPENFED_TASK_INFO))

    def set_task_info(self, task_info: TaskInfo):
        self.set(OPENFED_TASK_INFO, task_info.info_dict)

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
        super().register_hook(collector.bounding_name, collector)

    def collect(self):
        """Collect message from other side.
        """
        # read all collection information
        # key = None will return the whole info dictionary.
        info = self.get(None)
        for key, value in info.items():
            if key.startswith("Collector"):
                # don't forget () operation.
                obj: Collector = Register(key, self)()
                if obj is not None and self.world.leader and obj.leader_collector or \
                        self.world.follower and obj.follower_collector:
                    if not obj.once_only or not obj.collected:
                        obj.load_message(value)
                        logger.debug(obj)

    def scatter(self):
        """Scatter self.hook information to the other end.
        """
        cdict = {}
        for k, f in self.hook_dict.items():
            if self.world.leader and f.leader_scatter or\
                    self.world.follower and f.follower_scatter:
                if not f.once_only or not f.scattered:
                    cdict[k] = f()
        self._update(cdict)

    def register_cypher(self, cypher: Cypher) -> None:
        """Register a cypher to encrypt/decrypt the Tensor.
        """
        self.register_hook(cypher)

    def key_tensor_map(self, key: str, tensor: Tensor) -> None:
        """Add a new <key, tensor> pair to package.
        """
        if key in self.key_tensor_bidict or key == "param":
            raise KeyError(f"{key} already existed.")
        self.key_tensor_bidict[key] = tensor
        self.packages[key]["param"] = tensor

    def set_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Add a state_dict to package.
        """
        [self.key_tensor_map(k, v) for k, v in state_dict.items()]

    def reset_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Call reset() and set_state_dict() in a single step.
        """
        self.reset()
        self.set_state_dict(state_dict)

    def key_name(self, t: Tensor) -> str:
        """Return the string name for the given tensor t.
        """
        return self.key_tensor_bidict.inverse[t]

    def key_tensor(self, key: str) -> Tensor:
        """Return the tensor for the given key.
        """
        return self.key_tensor_bidict[key]

    def pack(self, key: Union[str, Tensor], rdict: Dict[str, Tensor]) -> None:
        """Update rdict to the key in package.
        """
        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        package = self.packages[key]

        package.update(rdict)

    def unpack(self, key: Union[str, Tensor], rdict: Dict[str, Any]) -> Dict[str, Tensor]:
        """Update rdict with the one saved in package.
        """
        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        package = self.packages[key]
        rdict = {k: package[k] for k in rdict}

        return rdict

    @property
    def tensor_indexed_packages(self) -> Dict[Tensor, Dict[str, Tensor]]:
        """Return a Dict which indexed by Tensor.
        """
        return {self.key_tensor(k): v for k, v in self.packages.items()}

    def reset(self) -> None:
        """Reset key_tensor_bidict and packages.
        """
        self.key_tensor_bidict = bidict()
        self.packages = defaultdict(dict)

    def pull(self, auto_load_param: bool = True) -> Union[Dict[str, Dict[str, Tensor]], Tuple[Work, Callable]]:
        """Pull data from the other end. 
        After received data, Follower will load `param` to Tensor by an in-place operation automatically.
        You can specify :param:auto_load_param as ``False`` to disable it.
        """
        assert self.country._get_group_size(
            self.pg) == 2, "Delivery is only designed for group with size 2"

        received = [None, None]

        rank = self.leader_rank if self.world.leader else self.follower_rank
        other_rank = self.follower_rank if self.world.leader else self.leader_rank

        def _op_after_gather(*args):
            r_packages: Dict = received[other_rank] # type: ignore

            # NOTE: decrypt data in the reverse order.
            for hook in self.hook_list[::-1]:
                r_packages = {k: hook.decrypt(self.key_tensor(k), v)
                              for k, v in r_packages.items()}

            # Follower will load `param` to Tensor by an in-place operation.
            if auto_load_param and self.world.follower:
                for k, v in r_packages.items():
                    if 'param' in v:
                        self.key_tensor_bidict[k].data.copy_(v['param'])
            self.packages = r_packages
            return r_packages

        returns = gather_object(
            None, received,
            dst=rank,
            group=self.pg,
            async_op=ASYNC_OP.is_async_op,
            country=self.country,
            global_rank=False)

        if ASYNC_OP.is_async_op:
            handler, step_func = returns # type: ignore
            # lambda: before go into this layer's function, call step_func first.
            return handler, lambda: _op_after_gather(step_func())
        else:
            return _op_after_gather()

    def push(self) -> Union[Tuple[Work, Callable], Any]:
        """Push data to the other end.
        """
        assert self.country._get_group_size(
            self.pg) == 2, "Delivery is only designed for group with size 2"

        rank = self.follower_rank if self.world.leader else self.leader_rank

        # encrypt data
        for hook in self.hook_list:
            self.packages = {k: hook.encrypt(self.key_tensor(k), v)
                             for k, v in self.packages.items()}

        return gather_object(
            self.packages, None,
            dst=rank,
            group=self.pg,
            async_op=ASYNC_OP.is_async_op,
            country=self.country,
            global_rank=False)

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Delivery",
            description=tablist(
                head=["Nick Name", "Upload Version",
                      "Download Version", "Status"],
                data=[self.nick_name, self.upload_version,
                      self.download_version, self._get_state()],
            )
        )
