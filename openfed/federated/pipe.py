# @Author            : FederalLab
# @Date              : 2021-09-25 16:52:34
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:52:34
# Copyright (c) FederalLab. All rights reserved.
import json
import time
import warnings
from datetime import timedelta
from typing import Any, Dict, Optional

import torch.distributed.distributed_c10d as distributed_c10d

from openfed.common import Meta
from openfed.utils import FMT, tablist
from .const import (aggregator, aggregator_rank, collaborator,
                    collaborator_rank, nick_name, offline, openfed_identity,
                    openfed_meta, openfed_status, pull, push, zombie)
from .exceptions import DeviceOffline
from .props import DistributedProperties, FederatedProperties


def set_store_value(store, key, value) -> bool:
    r"""Sets store value safely.
    """
    json_str = json.dumps(value)
    try:
        store.set(key, json_str)
    except Exception as e:
        warnings.warn(f'Set store value failed. {e}')
        return False
    finally:
        return True


def get_store_value(store, key) -> Any:
    r"""Gets store value safely.
    """
    try:
        bytes = store.get(key)
    except Exception as e:
        warnings.warn(f'Get store value failed. {e}')
        bytes = b''
    json_str = str(bytes, encoding='utf-8')
    return json.loads(json_str)


class Pipe():
    r'''Transfers data between nodes.

    Args:
        store: A TCP/FILE store to transfer message.
        pg: A Process Group to transfer tensor via different backend, such as
            `gloo`, `mpi`.
        dist_props: The distributed properties.
        fed_props: The federated properties.
    '''
    store: Any
    pg: Any
    dist_props: DistributedProperties
    fed_props: FederatedProperties

    _u_backup_info: Dict[str, Any]
    _i_backup_info: Dict[str, Any]

    read_successfully: bool

    @property
    def _i_key(self):
        return openfed_identity + '_' + self.role

    @property
    def _u_key(self):
        return openfed_identity + '_' + self.anti_role

    def _read(self, key: Optional[str] = None):
        read_info = get_store_value(self.store, self._u_key)
        if len(read_info) == 0:
            self.read_successfully = False
            self._u_backup_info[
                openfed_status] = offline if self.collaborator else zombie
        else:
            self.read_successfully = True
            self._u_backup_info = read_info
        return self._u_backup_info[key] if key else self._u_backup_info

    def _write(self, info: Dict) -> bool:
        info['timestep'] = time.time()
        return set_store_value(self.store, self._i_key, info)

    def _update(self, info: Dict) -> bool:
        self._i_backup_info.update(info)
        return self._write(self._i_backup_info)

    def get(self, key):
        r"""Get key value from self._u_key 's dictionary.
        If key is missed, will raise error.
        """
        return self._read(key)

    def direct_get(self, key):
        r"""Get key value directly from store.
        If key is missed, will wait until it has been set.
        """
        return get_store_value(self.store, key)

    def set(self, key: str, value):
        r"""Set key value to self._i_key 's dictionary.
        """
        self._update({key: value})

    def direct_set(self, key: str, value):
        r"""Set key value directly to store.
        """
        set_store_value(self.store, key, value)

    @property
    def role(self):
        return self.fed_props.role

    @property
    def anti_role(self):
        return aggregator if self.collaborator else collaborator

    @property
    def aggregator(self):
        return self.role == aggregator

    @property
    def collaborator(self):
        return self.role == collaborator

    @property
    def nick_name(self) -> Any:
        return self.get(nick_name)

    def __init__(
        self,
        store: Any,
        pg: Any,
        dist_props: DistributedProperties,
        fed_props: FederatedProperties,
    ):
        self.store = store
        self.pg = pg
        self.dist_props = dist_props
        self.fed_props = fed_props

        set_store_value(
            store=self.store,
            key=self._i_key,
            value={
                openfed_status: zombie,
                openfed_meta: Meta(),
                nick_name: self.fed_props.nick_name,
            })

        self._i_backup_info = get_store_value(self.store, self._i_key)
        self._u_backup_info = get_store_value(self.store, self._u_key)

        self.read_successfully = True

    def upload(self, data: Any):
        self.transfer(True, data)

    def download(self) -> Any:
        return self.transfer(False)

    def _get_state(self):
        return self.get(openfed_status)

    def _set_state(self, state):
        return self.set(openfed_status, state)

    def pulling(self):
        self._set_state(pull)

    @property
    def is_pulling(self) -> bool:
        return self._get_state() == pull

    def pushing(self):
        self._set_state(push)

    @property
    def is_pushing(self) -> bool:
        return self._get_state() == push

    def zombie(self):
        self._set_state(zombie)

    @property
    def is_zombie(self) -> bool:
        return self._get_state() == zombie

    def offline(self):
        self._set_state(offline)

    @property
    def is_offline(self) -> bool:
        return self._get_state() == offline

    def transfer(self, to: bool, data: Optional[Any] = None) -> Any:
        if self.is_offline:
            raise DeviceOffline(self)

        def _state():
            return self.is_pulling if to else self.is_pushing

        if self.collaborator:
            if to:
                self.pushing()
            else:
                self.pulling()

            tic = time.time()
            while not _state():
                if self.is_offline:
                    raise DeviceOffline(self)
                toc = time.time()
                if timedelta(seconds=toc - tic) > timedelta(minutes=30):
                    raise DeviceOffline(self)
        else:
            if not _state():
                raise DeviceOffline(self)
            else:
                if to:
                    self.pushing()
                else:
                    self.pulling()

        if to:
            self.push(data)
        else:
            data = self.pull()

        self.zombie()

        return data

    def push(self, data):
        assert distributed_c10d._get_group_size(self.pg) == 2,\
            'Pipe is only designed for point to point communication.'

        rank = collaborator_rank if self.aggregator else aggregator_rank
        rank = distributed_c10d._get_global_rank(self.pg, rank) \
            if distributed_c10d.get_world_size() > 2 else rank

        distributed_c10d.gather_object(data, None, dst=rank, group=self.pg)

    def pull(self) -> Any:
        assert distributed_c10d._get_group_size(self.pg) == 2,\
            'Pipe is only designed for point to point communication.'

        world_size = distributed_c10d.get_world_size()

        received = [None for _ in range(world_size)]

        rank = aggregator_rank if self.aggregator else collaborator_rank
        other_rank = collaborator_rank if self.aggregator else aggregator_rank

        rank = distributed_c10d._get_global_rank(
            self.pg, rank) if world_size > 2 else rank
        other_rank = distributed_c10d._get_global_rank(
            self.pg, other_rank) if world_size > 2 else other_rank

        distributed_c10d.gather_object(None, received, dst=rank, group=self.pg)

        data = [r for r in received if r is not None][0]
        assert data

        return data

    @property
    def meta(self) -> Meta:
        meta_dict = self.get(openfed_meta)
        assert self.read_successfully, 'read meta info failed'
        return Meta(**meta_dict)

    def set_meta(self, meta: Meta):
        self.set(openfed_meta, meta)

    def __del__(self):
        self.offline()

        def callback():
            distributed_c10d.destroy_process_group(self.pg)

            if distributed_c10d._group_count == 1:
                distributed_c10d.destroy_process_group()

        if not self.dist_props.lock.locked():
            with self.dist_props:
                callback()
        else:
            callback()

    def __repr__(self):
        head = ['nick name', 'status']
        data = [self.nick_name, self._get_state()]
        description = tablist(head, data, force_in_one_row=True)

        other_description = 'dist props: \n' + str(self.dist_props) + \
            'fed props: \n ' + str(self.fed_props)
        return FMT.openfed_class_fmt.format(
            class_name=self.__class__.__name__,
            description=description + '\n' + other_description)
