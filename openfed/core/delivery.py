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

import openfed
from bidict import bidict
from openfed.common import (Address, ArrayDict, ConnectTimeout,
                            DeviceOffline, Hook, InvalidAddress,
                            InvalidStoreReading, InvalidStoreWriting, Package,
                            TaskInfo, load_address_from_file, logger, peeper)
from openfed.hooks.collector import Collector, GPUInfo, Recorder, SystemInfo
from openfed.hooks.cypher import Cypher, FormatCheck
from openfed.utils import (convert_to_list, openfed_class_fmt, tablist,
                           time_string)
from random_words import RandomWords
from torch import Tensor
from torch._C._distributed_c10d import Work

from .const import *
from .functional import gather_object
from .space import (Country, ProcessGroup, Store, World, add_mt_lock,
                    del_mt_lock)

rw = RandomWords()

peeper.delivery_dict = ArrayDict()


def safe_store_set(store: Store, key: str, value: Dict) -> bool:
    jsonstr = json.dumps(value)
    try:
        store.set(key, jsonstr)
    except Exception as e:
        raise InvalidStoreWriting(e)
    return True


def safe_store_get(store: Store, key: str) -> Dict[str, Any]:
    try:
        jsonbytes = store.get(key)
    except Exception as e:
        raise InvalidStoreReading(e)
    jsonstr = str(jsonbytes, encoding='utf-8')
    return json.loads(jsonstr)


def fresh_read(func):
    """A decorate function that will raise error if the data is refresh.
    """

    def _fresh_read(self, *args, **kwargs):
        output = func(self, *args, **kwargs)
        if not self.fresh_read:
            logger.debug(
                "Use an cached value instead a fresh required data."
                "Which may cause Error."
                f"func: {func}"
                f"args: {args}"
                f"kwargs: {kwargs}")
        return output
    return _fresh_read


class Delivery(Hook, Package):
    """Contains all communication functions in Delivery.
    """
    world  : World
    country: Country
    pg     : ProcessGroup
    store  : Store

    # handler, step function, timestamp
    _download_hang_up: Tuple[Work, Callable, int]
    _upload_hang_up  : Tuple[Work, Callable, int]

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
        self.pg      = pg
        self.store   = store
        self.country = country
        self.world   = country.world

        # set nick name first!
        if self.world.leader:
            safe_store_set(
                store = self.store,
                key   = nick_name,
                value = rw.random_word()
            )
        self.nick_name = safe_store_get(
            store = self.store,
            key   = nick_name
        )

        # write self._i_key to initialize the key value store.
        safe_store_set(
            store = self.store,
            key   = self._i_key,
            value = {
                openfed_status   : zombie,
                openfed_task_info: TaskInfo(),
            }
        )

        # Fetch data at last
        # try to read _u_key from the other end to make sure it is online.
        self._i_backup_info = safe_store_get(self.store, self._i_key)
        self._u_backup_info = safe_store_get(self.store, self._u_key)

        # register a default collector
        self.register_collector(SystemInfo())
        self.register_collector(GPUInfo())

        self.scatter()

        # Run at the initialize state.
        self.collect()

        self.fresh_read        = True
        self.key_tensor_bidict = bidict()
        self.packages          = defaultdict(dict)
        
        self.register_cypher(FormatCheck())

        self.download_hang_up: bool = False
        self.upload_hang_up  : bool = False

        peeper.delivery_dict[self] = time_string()

    @property
    def upload_version(self) -> int:
        return self.get('upload_version')

    @property
    def download_version(self) -> int:
        return self.get("download_version")

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
        assert not self.is_offline, DeviceOffline(self)

        def _state():
            return self.is_pulling if to else self.is_pushing

        # logic judge
        if self.world.follower:
            # set state first
            [self.pushing() if to else self.pulling()]

            # wait until satisfied
            tic = time.time() if not tic else tic
            while not _state():
                assert not self.is_offline, DeviceOffline(self)
                toc = time.time()
                if timedelta(seconds=toc-tic) > timedelta(minutes=30):
                    raise ConnectTimeout(self)
                time.sleep(0.1)
        else:
            # check state first
            if not _state():
                return False
            else:
                [self.pushing() if to else self.pulling()]

        # transfer
        if handler:
            handler.wait()
            if step_func is not None:
                step_func()
        else:
            [self.push() if to else self.pull()]

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

        if not self.transfer(
            to        = to,
            handler   = handler,
            tic       = tic,
            step_func = step_func):
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

    def upload(self, version: int) -> bool:
        """Upload packages date to the other end.
        """

        # set version on task info
        self.set("upload_version", version)

        if self.world.async_op:
            handle, step_func = self.push()
            # store the necessary message, and hang up begining time.
            self._upload_hang_up = (  # type: ignore
                handle, step_func, time.time())
            self.upload_hang_up = True
            return False
        else:
            return self.transfer(to=True)

    def download(self, version: int) -> bool:
        """Download packages from other end.
        """

        # set version
        self.set("download_version", version)

        if self.world.async_op:
            handle, step_func = self.pull()
            self._download_hang_up = (  # type: ignore
                handle, step_func, time.time())
            self.download_hang_up = True
            return False
        else:
            return self.transfer(to=False)

    @classmethod
    def delivery_generator(cls) -> Any:
        """Return a generator to iterate over all delivery.
        """
        for delivery, _ in peeper.delivery_dict:
            yield [] if delivery is None else delivery
            if delivery is not None:
                delivery.world.current_pg = delivery.pg
        else:
            return []

    @classmethod
    def default_delivery(cls) -> Any:
        """Return the only delivery. If more then one, raise warning.
        """
        for delivery, _ in peeper.delivery_dict:
            return delivery

    @property
    def _i_key(self) -> str:
        return openfed_identity + "_" + ("LEADER" if self.world.leader else "FOLLOWER")

    @property
    def _u_key(self) -> str:
        return openfed_identity + "_" + ("LEADER" if not self.world.leader else "FOLLOWER")

    def _write(self, info: Dict[str, str]) -> bool:
        """Write info to self._i_key.
        """
        info["timestemp"] = time_string()
        return safe_store_set(self.store, self._i_key, info)

    def _update(self, info: Dict[str, str]) -> bool:
        """rewrite the old message in kv-store.
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
        self._update({key: value})

    def get(self, key: Union[None, str]) -> Any:
        return self._read(key)

    @property
    @fresh_read
    def task_info(self) -> TaskInfo:
        return TaskInfo(**self.get(openfed_task_info))

    def set_task_info(self, task_info: TaskInfo):
        self.set(openfed_task_info, task_info)

    def _get_state(self) -> str:
        return self.get(openfed_status)

    def _set_state(self, state: str):
        self.set(openfed_status, state)

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
        rdict   = {k: package[k] for k in rdict}

        return rdict

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
            self.pg) == 2, "Delivery is only designed for group with size 2"

        received = [None for _ in range(self.country.get_world_size())]

        rank       = leader_rank if self.world.leader else follower_rank
        other_rank = follower_rank if self.world.leader else leader_rank

        rank       = self.country._get_global_rank(self.pg, rank)
        other_rank = self.country._get_global_rank(self.pg, other_rank)

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
            self.pg) == 2, "Delivery is only designed for group with size 2"

        rank = follower_rank if self.world.leader else leader_rank
        rank = self.country._get_global_rank(self.pg, rank)

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
            class_name="Delivery",
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
    def destroy_delivery(cls, delivery: Delivery):
        world   = delivery.world
        pg      = delivery.pg
        country = delivery.country

        if pg == world.current_pg:
            world.current_pg = NULL_PG

        delivery.offline()
        del world._delivery_dict[delivery]
        del peeper.delivery_dict[delivery]

        country.destroy_process_group(pg)

        if country._group_count == 1:
            # If the country contains many deliveries, the group_count should be larger than 1
            # after delete a delivery. If equals to one, it means that only the world group is left.
            # So, we need to delete it manually.
            country.destroy_process_group()

    @classmethod
    def destroy_all_deliveries(cls):
        for delivery, _ in peeper.delivery_dict:
            cls.destroy_delivery(delivery)


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

    def run(self) -> str:
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
            delivery = Delivery(
                store   = country.get_store(sub_pg),
                pg      = sub_pg,
                country = country)
            with self.world._delivery_dict:
                self.world._delivery_dict[delivery] = time_string()
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
                 max_try_times   : int      = 5,
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

        self.max_try_times    = max_try_times
        self.interval_seconds = interval_seconds

        add_mt_lock(self)

        self.world = world

        self.read_address_from_file()

        self.stopped = False

        if self.world.leader:
            self.start()
            if not openfed.DAL.is_dal:
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
        while not self.stopped and self.world.ALIVE:
            # update pending list
            self.read_address_from_file()
            # Create new Joint for new address
            for address, (last_time, try_times) in self.pending_queue:
                if address not in joint_map:
                    joint_map[address] = Joint(address, self.world)

            def try_now(last_time, try_times) -> bool:
                return False if (time.time() - last_time < self.interval_seconds) or try_times >= self.max_try_times else True

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
                        if try_times > self.max_try_times:
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

            if openfed.DAL.is_dal:
                time.sleep(self.interval_seconds)
            else:
                break

        self.abnormal_exited = len(
            self.pending_queue) > 0 or len(self.discard_queue) > 0
        return f"Build connection to {len(self.finished_queue)} addresses."

    def kill_world(self) -> None:
        self.world.kill()

    def manual_stop(self, kill_world: bool = True) -> None:
        if kill_world:
            self.kill_world()
        del_mt_lock(self)
        self.stopped = True

    def manual_joint(self, address: Address) -> None:
        if not openfed.DAL.is_dal and self.world.leader:
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
