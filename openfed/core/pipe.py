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
from subprocess import call
import time
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, Union

import torch.distributed.distributed_c10d as distributed_c10d
from bidict import bidict
from openfed.common import (Attach, ConnectTimeout, DeviceOffline,
                            InvalidStoreReading, InvalidStoreWriting, Package,
                            TaskInfo, logger, peeper)
from openfed.hooks.collector import Collector, GPUInfo, Recorder, SystemInfo
from openfed.hooks.cypher import Cypher, DeviceAlign, FormatChecker
from openfed.utils import openfed_class_fmt, tablist, time_string
from random_words import RandomWords
from torch import Tensor
from torch.distributed import ProcessGroup, Store, gather_object

from .const import *
from .federated import DistributedProperties, FederatedGroupProperties

rw = RandomWords()


def safe_store_set(store: Store, key: str, value: Dict) -> bool:
    """Write key to store safely.
    """
    jsonstr = json.dumps(value)
    try:
        store.set(key, jsonstr)
    except Exception as e:
        raise InvalidStoreWriting(e)
    return True


def safe_store_get(store: Store, key: str) -> Dict[str, Any]:
    """Read key from state safely.
    """
    try:
        jsonbytes = store.get(key)
    except Exception as e:
        raise InvalidStoreReading(e)
    jsonstr = str(jsonbytes, encoding='utf-8')
    return json.loads(jsonstr)


def fresh_read(func):
    """A decorate function that will raise error if the data is not fresh.
    If read key value from the store is failed, we will use the cached info.
    """
    def _fresh_read(self, *args, **kwargs):
        output = func(self, *args, **kwargs)
        assert self.fresh_read, "Data out of date."
        return output

    return _fresh_read


class Pipe(Attach, Package):
    """
    Pipe is responsible for transfer tensors and any other short information
    to the other hand. You can pack state dict of Aggregator, Pipe to Pipe. Vise via,
    You can unpack inner state from Pipe to Aggregator or Pipe.

    Pipe is the unified API that provided for user to access.

    .. warn::
        In federated learning, the tensor transfer between each node may be varied significantly.
        And it may be diffcult to maintain all the received data previously like Distributed Learing.
        (In distributed learning, each node will transfer certain tensor.)
        In Pipe, we will organize all data as a dictionary which using a unifed string name 
        across all nodes. Then, we use the `gather_object` method to transfer tensors. This also 
        requires that all the data transferred should be picklable.
    
    .. warn::
        Make sure you do any operation about Pipe is under the context of `pipe.distributed_properties`.
    """
    pg: ProcessGroup
    store: Store

    # Sometimes, the read operation will be failed for unknown reasons.
    # Essentially when we do simulation experiments on a single node.
    # In order to avoid this error, we always backup the last info
    # if failed, return this instead.
    _u_backup_info: Dict[str, Any]

    # you should ignore this dict. it will make you feel confused.
    _i_backup_info: Dict[str, Any]

    # indicates whether current data is
    fresh_read: bool

    # do not set this manually!
    _nick_name: str
    key_tensor_bidict: bidict
    packages: Dict[str, Dict[str, Tensor]]

    def __init__(
        self,
        store: Store,
        pg: ProcessGroup,
        distributed_properties: DistributedProperties,
        federated_group_properties: FederatedGroupProperties,
    ) -> None:
        """
        Args:
            store: It is a long-time connection, we can transfer information
                with other end.
            pg: The process group belong to. In fact, the process group always
                contains two process, the leader and follower.
            country: The pg belongs to. Country also contains all global variables
                that shared among different pipe.
        """
        self.pg = pg
        self.store = store

        self.distributed_properties = distributed_properties
        self.federated_group_properties = federated_group_properties

        # Set nick name on leader end.
        # Nick name is always assigned by leader.
        # With this nick name, we can have a better identification
        # of each follower.
        # warn: the nick name may be not unique.
        if self.leader:
            safe_store_set(store=self.store,
                           key=nick_name,
                           value=rw.random_word())
        # Record the nick name.
        # So that we can avoid to read it from store every time.
        self.nick_name = safe_store_get(store=self.store, key=nick_name)

        # Write self._i_key to initialize the key value store.
        safe_store_set(store=self.store,
                       key=self._i_key,
                       value={
                           openfed_status: zombie,
                           openfed_task_info: TaskInfo(),
                       })

        # Try to read _u_key from the other end.
        # This is also a lock to make sure that the other end is online.
        self._i_backup_info = safe_store_get(self.store, self._i_key)
        self._u_backup_info = safe_store_get(self.store, self._u_key)

        # Register default collectors
        self.register_collector(SystemInfo())
        self.register_collector(GPUInfo())

        # Scatter will call collector and set them in the store.
        self.scatter()

        # Collect is the inverse process of Scatter, which will read other
        # hand information then store it in self collector.
        self.collect()

        self.fresh_read = True

        # A bi-direction map that bounding the name with the tensor.
        self.key_tensor_bidict = bidict()
        # Packages use the string name as dictionary.
        self.packages = defaultdict(dict)

        # FormatChecker is a necessary cypher, which will move the tensor to
        # CPU, and then convert back to corresponding GPU.
        self.register_cypher(FormatChecker())
        self.register_cypher(DeviceAlign())

    @property
    def leader(self):
        return self.role == leader

    @property
    def follower(self):
        return self.role == follower

    @property
    def role(self):
        return self.federated_group_properties.role

    @property
    def upload_version(self) -> int:
        return self.get('upload_version')

    @property
    def download_version(self) -> int:
        return self.get("download_version")

    def transfer(self, to: bool) -> bool:
        """
        Args:
            to: If `True`, transfer data to other end. Otherwise, download 
                data from the other end.
        """
        if self.is_offline: raise DeviceOffline(self)

        def _state():
            return self.is_pulling if to else self.is_pushing

        # logic judge
        if self.follower:
            # set state first
            [self.pushing() if to else self.pulling()]

            # wait until satisfied
            tic = time.time()
            while not _state():
                if self.is_offline: raise DeviceOffline(self)
                toc = time.time()
                if timedelta(seconds=toc - tic) > timedelta(minutes=30):
                    raise ConnectTimeout(self)
        else:
            # check state first
            if not _state():
                return False
            else:
                [self.pushing() if to else self.pulling()]

        # Fetch task info
        [self.push() if to else self.pull()]

        self.zombie()
        return True

    def set_upload_version(self, version: int):
        self.set("upload_version", version)

    def upload(self, version: int) -> bool:
        """Upload packages date to the other end.
        Args:
            version: The version of current upload packages.
        """

        # set version on task info
        self.set_upload_version(version)
        return self.transfer(to=True)

    def set_download_version(self, version: int):
        self.set("download_version", version)

    def download(self, version: int) -> bool:
        """Download packages from other end.
        Args:
            version: The version requested by current process.
        """

        # set version
        self.set_download_version(version)

        return self.transfer(to=False)

    @classmethod
    def pipe_generator(cls) -> Any:
        """Return a generator to iterate over all deliveries.
        """
        for pipe, _ in peeper.pipe_dict:  # type: ignore
            yield [] if pipe is None else pipe
            if pipe is not None:
                pipe.world.current_pg = pipe.pg
        else:
            return []

    @classmethod
    def default_pipe(cls) -> Any:
        """Return the fist pipe.
        """
        for pipe, _ in peeper.pipe_dict:  # type: ignore
            return pipe

    @property
    def _i_key(self) -> str:
        """Pipe will write information to `i_key`.
        """
        return openfed_identity + "_" + ("LEADER"
                                         if self.leader else "FOLLOWER")

    @property
    def _u_key(self) -> str:
        """Pipe will read information from `u_key`.
        """
        return openfed_identity + "_" + ("LEADER"
                                         if not self.leader else "FOLLOWER")

    def _write(self, info: Dict[str, str]) -> bool:
        """Write info to self._i_key.

        .. warn::
            This will erase all information directly.
        """
        info["timestemp"] = time_string()
        return safe_store_set(self.store, self._i_key, info)

    def _update(self, info: Dict[str, str]) -> bool:
        """Update the value in store with info.
        """
        # read i_key information, then update it
        self._i_backup_info.update(info)
        return self._write(self._i_backup_info)

    def _read(self, key: str = None) -> Dict:
        """Read message from self._u_key.
        """
        try:
            self._u_backup_info = safe_store_get(self.store, self._u_key)
            self.fresh_read = True
        except InvalidStoreReading as e:
            logger.debug(e)
            # use the cached one instead.
            # but at the same time, we need to set the state as zombie
            # otherwise the last state value may make the progress get stuck.
            # The server is quiet stable, if read failed, we think it is offline.
            # But client sometimes may be unstable, if read failed, we will assume it
            # go into offline.
            self._u_backup_info[
                openfed_status] = offline if self.follower else zombie
            self.fresh_read = False
        finally:
            return self._u_backup_info[key] if key else self._u_backup_info

    def set(self, key: str, value: Any):
        """Set key-value to store.
        """
        self._update({key: value})

    def get(self, key: Union[None, str]) -> Any:
        """Get key from store.
        """
        return self._read(key)

    @property
    @fresh_read
    def task_info(self) -> TaskInfo:
        """Get task info from store.
        The task infomation must be fresh read.
        """
        return TaskInfo(**self.get(openfed_task_info))

    def set_task_info(self, task_info: TaskInfo):
        """Set task infomation to the store directly.
        """
        self.set(openfed_task_info, task_info)

    def _get_state(self) -> str:
        """Return the other end state.
        """
        return self.get(openfed_status)

    def _set_state(self, state: str):
        """Set self state.
        """
        self.set(openfed_status, state)

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
        return self._get_state() == offline

    @Recorder.add_to_pool
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
                obj: Collector = Recorder(key, self)()
                if obj is not None:
                    if (self.leader and obj.leader_collector) or \
                        (self.follower and obj.follower_collector):
                        if not obj.once_only or not obj.collected:
                            obj.load_message(value)
                            logger.debug(obj)

    def scatter(self):
        """Scatter self.hook information to the other end.
        """
        cdict = {}
        for k, f in self.hook_dict.items():
            if self.leader and f.leader_scatter or\
                    self.follower and f.follower_scatter:
                if not f.once_only or not f.scattered:
                    cdict[k] = f()
        self._update(cdict)

    def register_cypher(self, cypher: Cypher) -> None:
        """Recorder a cypher to encrypt/decrypt the Tensor.
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

    def reset(self) -> None:
        """Reset key_tensor_bidict and packages.
        """
        self.key_tensor_bidict.clear()
        self.packages.clear()

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
        Args:
            key: The key index, both tensor and string is acceptable.
            rdict: The dictionary contains state information. (string indexed)

        .. warn::
            We will automatically skip the key with `None` value. (Set state with
            `None`, may lead to some error at latter process.)
        """
        if not rdict:
            return

        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        # Filter out the `None` value.
        rdict = {k: v for k, v in rdict.items() if v is not None}
        self.packages[key].update(rdict)

    def unpack(self, key: Union[str, Tensor],
               rdict: Dict[str, Any]) -> Dict[str, Tensor]:
        """Update rdict with the one saved in package.
        Args:
            key: The key index, both tensor and string is acceptable.
            rdict: The dictionary contains required state information.

        .. warn::
            Sometimes, the required key in rdict may be missing, we will not raise
            any expection or drop any error, but just skip those keys. So, be careful
            with the returned dictionary.
        """
        if not rdict:
            return rdict

        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        package = self.packages[key]
        return {k: package[k] for k in rdict if k in package}

    @property
    def tensor_indexed_packages(self) -> Dict[Tensor, Dict[str, Tensor]]:
        """Return a Dict which indexed by Tensor.
        """
        return {self.key_tensor(k): v for k, v in self.packages.items()}

    def pull(self, auto_load_param: bool = True) -> None:
        """Pull data from the other end. 
        After received data, Follower will load `param` to Tensor by an in-place operation automatically.
        You can specify :param:auto_load_param as ``False`` to disable it.
        """
        assert distributed_c10d._get_group_size(
            self.pg) == 2, "Pipe is only designed for group with size 2"

        world_size = distributed_c10d.get_world_size()

        received = [None for _ in range(world_size)]

        rank = leader_rank if self.leader else follower_rank
        other_rank = follower_rank if self.leader else leader_rank

        rank = distributed_c10d._get_global_rank(
            self.pg, rank) if world_size > 2 else rank
        other_rank = distributed_c10d._get_global_rank(
            self.pg, other_rank) if world_size > 2 else other_rank

        gather_object(None, received, dst=rank, group=self.pg)

        r_packages = [r for r in received if r is not None][0]
        assert r_packages is not None

        # NOTE: decrypt data in the reverse order.
        for hook in self.hook_list[::-1]:
            r_packages = {
                k: hook.decrypt(self.key_tensor(k), v)
                for k, v in r_packages.items()
            }

        # Follower will load `param` to Tensor by an in-place operation.
        if auto_load_param and self.follower:
            for k, v in r_packages.items():
                if 'param' in v:
                    self.key_tensor_bidict[k].data.copy_(v['param'])
        self.packages.update(r_packages)

    def push(self) -> None:
        """Push data to the other end.
        """
        assert distributed_c10d._get_group_size(
            self.pg) == 2, "Pipe is only designed for group with size 2"

        rank = follower_rank if self.leader else leader_rank
        rank = distributed_c10d._get_global_rank(
            self.pg, rank) if distributed_c10d.get_world_size() > 2 else rank

        # encrypt data
        packages = self.packages
        for hook in self.hook_list:
            packages = {
                k: hook.encrypt(self.key_tensor(k), v)
                for k, v in packages.items()
            }

        gather_object(packages, None, dst=rank, group=self.pg)

    def __str__(self) -> str:
        return openfed_class_fmt.format(class_name="Pipe",
                                        description=tablist(
                                            head=[
                                                "Nick Name", "Upload Version",
                                                "Download Version", "Status"
                                            ],
                                            data=[
                                                self.nick_name,
                                                self.upload_version,
                                                self.download_version,
                                                self._get_state()
                                            ],
                                        ))

    def __del__(self):
        self.offline()

        def callback():
            distributed_c10d.destroy_process_group(self.pg)

            if distributed_c10d._group_count == 1:
                distributed_c10d.destroy_process_group()

        if self.distributed_properties.lock.locked():
            with self.distributed_properties:
                callback()
        else:
            callback()
