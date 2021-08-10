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
from threading import Lock, Thread
from typing import Any, Callable, Dict, Tuple, Union

from bidict import bidict
from openfed.common import (Address, ArrayDict, Attach, ConnectTimeout,
                            DeviceOffline, InvalidAddress, InvalidStoreReading,
                            InvalidStoreWriting, Package, TaskInfo,
                            load_address_from_file, logger, peeper)
from openfed.hooks.collector import Collector, GPUInfo, Recorder, SystemInfo
from openfed.hooks.cypher import Cypher, FormatChecker
from openfed.utils import openfed_class_fmt, tablist, time_string
from random_words import RandomWords
from torch import Tensor
from torch._C._distributed_c10d import Work

from .const import *
from .functional import gather_object
from .space import (Country, ProcessGroup, Store, World, add_mt_lock,
                    del_mt_lock)

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

class DelayHandler(object):
    """An empty handler to skip the upload/download function. 
    It is useful when you don't want to upload the tensor but send
    the task info to other hand.
    """
    def __init__(self, func):
        self.func = func
        self.handler = None
        self.step_func = None
    
    def wait(self):
        if self.handler is None and self.step_func is None:
            self.handler, self.step_func = self.func()
        self.handler.wait() # type: ignore
        self.step_func() # type: ignore

    def is_completed(self) -> bool:
        return self.handler.is_completed() # type: ignore

class Pipe(Attach, Package):
    """Pipe is responsible for transfer tensors and any other short information
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
    """
    world  : World
    country: Country
    pg     : ProcessGroup
    store  : Store

    # handler, step function, timestamp
    _download_hang_up: DelayHandler
    _upload_hang_up  : DelayHandler

    download_hang_up: bool
    upload_hang_up  : bool

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
    _nick_name       : str
    key_tensor_bidict: bidict
    packages         : Dict[str, Dict[str, Tensor]]

    def __init__(self,
                 store  : Store,
                 pg     : ProcessGroup,
                 country: Country,
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
        self.pg      = pg
        self.store   = store
        self.country = country
        self.world   = country.world

        # Set nick name on leader end.
        # Nick name is always assigned by leader.
        # With this nick name, we can have a better identification
        # of each follower.
        # warn: the nick name may be not unique.
        if self.world.leader:
            safe_store_set(
                store = self.store,
                key   = nick_name,
                value = rw.random_word()
            )
        # Record the nick name.
        # So that we can avoid to read it from store every time.
        self.nick_name = safe_store_get(
            store = self.store,
            key   = nick_name
        )

        # Write self._i_key to initialize the key value store.
        safe_store_set(
            store = self.store,
            key   = self._i_key,
            value = {
                openfed_status   : zombie,
                openfed_task_info: TaskInfo(),
            }
        )

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

        # download and upload hang up indicate that whether it is dealling
        # with a download or upload event.
        # Only used while async is enable.
        self.download_hang_up: bool = False
        self.upload_hang_up  : bool = False

        # Register pipe to world and peeper dictionary
        self.world.register_delivery(self)

    @property
    def follower(self):
        return self.world.follower
    
    @property
    def leader(self):
        return self.world.leader

    @property
    def upload_version(self) -> int:
        return self.get('upload_version')

    @property
    def download_version(self) -> int:
        return self.get("download_version")

    def transfer(self,
                 to       : bool,
                 handler  : Union[DelayHandler, None] = None) -> bool: 
        """
        Args:
            to: If `True`, transfer data to other end. Otherwise, download 
                data from the other end.
            handler: If not None, call handler.wait(). Otherwise, call `pull()`
                or `push` directly.
            tic: start counting time.
        """
        if self.is_offline: raise DeviceOffline(self)

        def _state():
            return self.is_pulling if to else self.is_pushing

        # HACK: In order to reduce the unnecessary tensor transfer between 
        # follower and leader, we will not upload tensor if the model has not changed.
        # In other word, if you are under the test mode, it is unnecessary to upload
        # the unmodified tensor to leader. However, this feature improvement will
        # bring a new hack: HACK-0 will listen to the other end's state with a short
        # time sleep. At general case, leader will set the state and waiting to 
        # transfer data from follower. So, everything goes well.
        # But if we skip the download process, the state in leader will only be setted 
        # as pulling or pushing for quiet short time, that HACK-0 may miss the state 
        # change signal, and always keep wait. This hack will cause the error of 
        # `Invalid part ID`. (The leader read the task info again and again, but the 
        # follower is never able to capture the statue change.)
        # It is vital to add a sleep if leader do not transfer data at HACK-1 and HACK-2.

        # logic judge
        if self.world.follower:
            # set state first
            [self.pushing() if to else self.pulling()]

            # wait until satisfied
            tic = time.time()
            while not _state():
                if self.is_offline: raise DeviceOffline(self)
                toc = time.time()
                if timedelta(seconds=toc-tic) > timedelta(minutes=30):
                    raise ConnectTimeout(self)
                # HACK-0
                time.sleep(0.01)
        else:
            # check state first
            if not _state():
                return False
            else:
                [self.pushing() if to else self.pulling()]

        # Fetch task info
        train = self.task_info.mode == 'train' # type: ignore
        
        # transfer
        # Fake download/upload or real download/upload
        # 1. follower will always download the data
        # 2. follower will only upload data if train == True
        # 3. leader will always upload data
        # 4. leader will only download data if train == True
        def callback():
            if handler:
                handler.wait()
            else:
                [self.push() if to else self.pull()]

        if self.follower:
            if not to or train:
                callback()
            else:
                # HACK-1
                time.sleep(0.1)
        elif self.leader:
            if to or train:
                callback()
            else:
                # HACK-2
                time.sleep(0.1)

        self.zombie()
        return True

    def deal_with_hang_up(self) -> bool:
        """Dealing with the handler for hang up operations.
        """
        if self.upload_hang_up:
            handler = self._upload_hang_up
            to = True
        elif self.download_hang_up:
            handler = self._download_hang_up
            to = False
        else:
            return False

        if not self.transfer(
            to        = to,
            handler   = handler):
            return False

        if handler.is_completed():
            if self.upload_hang_up:
                del self._upload_hang_up
                self.upload_hang_up = False
            elif self.download_hang_up:
                del self._download_hang_up
                self.download_hang_up = False
            return True
        else:
            return False

    def set_upload_version(self, version: int):
        self.set("upload_version", version)

    def upload(self, version: int) -> bool:
        """Upload packages date to the other end.
        Args:
            version: The version of current upload packages.
        """

        # set version on task info
        self.set_upload_version(version)

        if self.world.async_op:
            # store the necessary message, and hang up begining time.
            self._upload_hang_up = DelayHandler(self.push)
            self.upload_hang_up = True
            return False
        else:
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

        if self.world.async_op:
            self._download_hang_up = DelayHandler(self.pull)
            self.download_hang_up = True
            return False
        else:
            return self.transfer(to=False)

    @classmethod
    def delivery_generator(cls) -> Any:
        """Return a generator to iterate over all deliveries.
        """
        for pipe, _ in peeper.delivery_dict: # type: ignore
            yield [] if pipe is None else pipe
            if pipe is not None:
                pipe.world.current_pg = pipe.pg
        else:
            return []

    @classmethod
    def default_delivery(cls) -> Any:
        """Return the fist pipe.
        """
        for pipe, _ in peeper.delivery_dict: # type: ignore
            return pipe

    @property
    def _i_key(self) -> str:
        """Pipe will write information to `i_key`.
        """
        return openfed_identity + "_" + ("LEADER" if self.world.leader else "FOLLOWER")

    @property
    def _u_key(self) -> str:
        """Pipe will read information from `u_key`.
        """
        return openfed_identity + "_" + ("LEADER" if not self.world.leader else "FOLLOWER")

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
            self.fresh_read     = True
        except InvalidStoreReading as e:
            logger.debug(e)
            # use the cached one instead.
            # but at the same time, we need to set the state as zombie
            # otherwise the last state value may make the progress get stuck.
            # The server is quiet stable, if read failed, we think it is offline.
            # But client sometimes may be unstable, if read failed, we will assume it
            # go into offline.
            self._u_backup_info[openfed_status] = offline if self.world.follower else zombie
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

    @property
    def alive(self) -> bool:
        """opposite to self.offline
        """
        return self.world.alive and self._get_state() != offline

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
        return self.world.alive and self._get_state() == offline

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
                    if (self.world.leader and obj.leader_collector) or \
                        (self.world.follower and obj.follower_collector):
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


    def unpack(self, key: Union[str, Tensor], rdict: Dict[str, Any]) -> Dict[str, Tensor]:
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

    def pull(self, 
        auto_load_param: bool = True) -> Union[Dict[str, Dict[str, Tensor]], Tuple[Work, Callable]]:
        """Pull data from the other end. 
        After received data, Follower will load `param` to Tensor by an in-place operation automatically.
        You can specify :param:auto_load_param as ``False`` to disable it.
        """
        assert self.country._get_group_size(
            self.pg) == 2, "Pipe is only designed for group with size 2"

        received = [None for _ in range(self.country.get_world_size())]

        rank       = leader_rank if self.world.leader else follower_rank
        other_rank = follower_rank if self.world.leader else leader_rank

        rank       = self.country._get_global_rank(self.pg, rank) if self.country.get_world_size() > 2 else rank
        other_rank = self.country._get_global_rank(self.pg, other_rank) if self.country.get_world_size() > 2 else other_rank

        def _op_after_gather(*args):
            r_packages = [r for r in received if r is not None][0]
            assert r_packages is not None

            # NOTE: decrypt data in the reverse order.
            for hook in self.hook_list[::-1]:
                r_packages = {k: hook.decrypt(self.key_tensor(k), v)
                              for k, v in r_packages.items()}

            # Follower will load `param` to Tensor by an in-place operation.
            if auto_load_param and self.world.follower:
                for k, v in r_packages.items():
                    if 'param' in v:
                        self.key_tensor_bidict[k].data.copy_(v['param'])
            self.packages.update(r_packages)
            return r_packages

        returns = gather_object(
            None,
            received,
            dst      = rank,
            group    = self.pg,
            async_op = self.world.async_op,
            country  = self.country)

        if self.world.async_op:
            handler, step_func = returns  # type: ignore
            # lambda: before go into this layer's function, call step_func first.
            return handler, lambda: _op_after_gather(step_func())
        else:
            return _op_after_gather()

    def push(self) -> Union[Tuple[Work, Callable], Any]:
        """Push data to the other end.
        """
        assert self.country._get_group_size(
            self.pg) == 2, "Pipe is only designed for group with size 2"

        rank = follower_rank if self.world.leader else leader_rank
        rank = self.country._get_global_rank(self.pg, rank) if self.country.get_world_size() > 2 else rank

        # encrypt data
        packages = self.packages
        for hook in self.hook_list:
            packages = {k: hook.encrypt(self.key_tensor(k), v)
                        for k, v in packages.items()}

        return gather_object(
            packages,
            None,
            dst      = rank,
            group    = self.pg,
            async_op = self.world.async_op,
            country  = self.country)

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Pipe",
            description=tablist(
                head=["Nick Name", 
                      "Upload Version",
                      "Download Version", 
                      "Status"],
                data=[self.nick_name, 
                      self.upload_version,
                      self.download_version, 
                      self._get_state()],
            )
        )


class Destroy(object):
    """Automatically destroy a process group along with its world.
    """
    @classmethod
    def destroy_delivery(cls, pipe: Pipe):
        world   = pipe.world
        pg      = pipe.pg
        country = pipe.country

        if pg == world.current_pg:
            world.current_pg = NULL_PG

        pipe.offline()
        # NOTE:
        # world._delivery_dict only recording the pipe defined under 
        # the same world. 
        # peeper.delivery_dict contains all defined pipe under all world.
        world.delete_delivery(pipe)
        
        country.destroy_process_group(pg)

        if country._group_count == 1:
            # If the country contains many deliveries, the group_count should be larger than 1
            # after delete a pipe. If equals to one, it means that only the world group is left.
            # So, we need to delete it manually.
            country.destroy_process_group()

    @classmethod
    def destroy_all_deliveries(cls):
        for pipe, _ in peeper.delivery_dict: # type: ignore
            cls.destroy_delivery(pipe)


class Joint(Thread):
    """A thread to build connection among specified ends.
    """

    # Indicates whether the connection is established correctly.
    build_success: bool

    def __init__(self, address: Address, world: World) -> None:
        super().__init__(daemon=True)

        if address.world_size == 2:
            # If this address is point to point access, set the correct rank for leader and follower.
            rank = follower_rank if world.follower else leader_rank
            address = address._replace(rank=rank)
        if address.world_size > 10 and address.init_method.startswith("tcp"):
            raise InvalidAddress(
                "Use `init_method=file:///tmp/openfed.sharefile` instead when `world_size > 10`.")

        self.address       = address
        self.build_success = False
        self.world         = world

        self.start()

        if self.world.follower:
            self.join()

    def run(self):
        # create a country
        country = Country(self.world)

        # build the connection between the country
        handler = country.init_process_group(*self.address)

        while not handler():
            time.sleep(0.01)

        # rank is always set to 0 for that we want to build a
        # point2point connection between the master and each nodes.
        # If the address is a point2point one, we should use the leader rank.
        # If the address is a shared multi-node one, we take the rank 0 as the leader rank.
        # And the re-arranged rank will be set to the ideal rank order in function call.
        sub_pg_list = country.build_point2point_group(
            rank=leader_rank if country.get_world_size() == 2 else 0)

        # bound pg with the country
        for sub_pg in sub_pg_list:
            # Pipe will automatically register to the corresponding world dictionary.
            Pipe(
                store   = country.get_store(sub_pg),
                pg      = sub_pg,
                country = country)
        self.build_success = True

class Maintainer(Thread):
    """
    Dynamic build the connection.
    """
    # unfinished address
    # Address -> [create time, try times]
    pending_queue: ArrayDict

    # finished address
    # Address -> [connection time, try times]
    finished_queue: Dict[Address, Tuple[float, int]]

    # discard address
    # Address -> [discarded time, try times]
    discard_queue: Dict[Address, Tuple[float, int]]

    mt_lock: Lock
    # The shared information among all country in this maintainer.
    world: World

    abnormal_exited: bool

    stopped: bool

    def __init__(self,
                 world           : World,
                 address         : Address = None,
                 address_file    : str      = None,
                 mtt   : int      = 5,
                 interval_seconds: float    = 10) -> None: 
        """
            Only a single valid address is allowed in client.
        """
        super().__init__(daemon=True)
        self.address_file = address_file
        self.pending_queue = ArrayDict()
        if address is not None:
            self.pending_queue[address] = [time.time(), 0]

        self.finished_queue  = dict()
        self.discard_queue   = dict()
        self.abnormal_exited = False

        self.mtt    = mtt
        self.interval_seconds = interval_seconds

        add_mt_lock(self)

        self.world = world

        self.read_address_from_file()

        self.stopped = False

        if self.world.leader:
            self.start()
            if not self.world.dal:
                self.join()
        else:
            assert len(self.pending_queue) == 1, "Only single address is allowed."
            address, (create_time, try_times) = self.pending_queue[0]
            Joint(address, self.world)
            with self.pending_queue:
                del self.pending_queue[address]
            self.finished_queue[address] = (time.time(), try_times+1)

    def read_address_from_file(self) -> None:
        address_list = load_address_from_file(self.address_file)

        for address in address_list:
            if address in self.pending_queue:
                # Already in pending queue.
                ...
            elif address in self.finished_queue:
                # Already in finished_queue.
                ...
            elif address in self.discard_queue:
                logger.error(f"Discarded!\nInvalid address: {address}.")
            else:
                # add address to pending queue
                with self.pending_queue:
                    self.pending_queue[address] = (time.time(), 0)

    def run(self) -> str:
        joint_map = dict()  # address -> joint
        while not self.stopped and self.world.alive:
            # update pending list
            self.read_address_from_file()
            # Create new Joint for new address
            for address, (last_time, try_times) in self.pending_queue:
                if address not in joint_map:
                    joint_map[address] = Joint(address, self.world)

            def try_now(last_time, try_times) -> bool:
                return False if (time.time() - last_time < self.interval_seconds) or try_times >= self.mtt else True

            rm_address = []
            for address, joint in joint_map.items():
                last_time, try_times = self.pending_queue[address]
                if try_now(last_time, try_times):
                    if joint.build_success:
                        self.finished_queue[address] = (
                            time.time(), try_times + 1)
                        with self.pending_queue:
                            del self.pending_queue[address]
                        rm_address.append((address))
                    else:
                        try_times += 1
                        if try_times > self.mtt:
                            # Stop and delete the joint
                            joint._stop()
                            rm_address.append(address)
                            # Move to discard_queue
                            self.discard_queue[address] = (
                                time.time(), try_times)
                            with self.pending_queue:
                                del self.pending_queue[address]
                            break
                        else:
                            with self.pending_queue:
                                self.pending_queue[address] = (
                                    time.time(), try_times)
            for address in rm_address:
                del joint_map[address]

            if self.world.dal:
                time.sleep(self.interval_seconds)
            else:
                break

        self.abnormal_exited = len(
            self.pending_queue) > 0 or len(self.discard_queue) > 0
        return f"Build connection to {len(self.finished_queue)} addresses."

    def manual_stop(self, kill_world: bool = True) -> None:
        """Kill current pipe as soon as possible.
        """
        if kill_world:
            self.world.kill()
        del_mt_lock(self)
        self.stopped = True

    def manual_joint(self, address: Address) -> None:
        if not self.world.dal and self.world.leader:
            raise RuntimeError("Dynamic Address Loading (ADL) is disabled.")

        if self.world.leader:
            with self.pending_queue:
                self.pending_queue[address] = (time.time(), 0)
        else:
            Joint(address, self.world)

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Maintainer",
            description=tablist(
                head=["Pending", 
                      "Finished", 
                      "Discard"],
                data=[len(self.pending_queue),
                      len(self.finished_queue),
                      len(self.discard_queue)],
                force_in_one_row=True,
            )
        )
