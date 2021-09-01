# Copyright (c) FederalLab. All rights reserved.
import json
import time
import warnings
from datetime import timedelta
from typing import Any, Dict, Optional

import torch.distributed.distributed_c10d as distributed_c10d
from openfed.common import DeviceOffline, TaskInfo
from openfed.utils import openfed_class_fmt, tablist, time_string
from torch.distributed import ProcessGroup, Store, gather_object

from .const import *
from .federated import DistributedProperties, FederatedProperties


def set_store_value(store, key, value) -> bool:
    json_str = json.dumps(value)
    try:
        store.set(key, json_str)
    except Exception as e:
        warnings.warn(f"Set store value failed. {e}")
        return False
    finally:
        return True


def get_store_value(store, key) -> Any:
    try:
        bytes = store.get(key)
    except Exception as e:
        warnings.warn(f"Get store value failed. {e}")
        bytes = b''
    json_str = str(bytes, encoding='utf-8')
    return json.loads(json_str)


class Pipe():

    store: Store
    pg: ProcessGroup
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
                openfed_status] = offline if self.follower else zombie
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
        return self._read(key)

    def set(self, key: str, value):
        self._update(dict(key=value))

    @property
    def role(self):
        return self.fed_props.role

    @property
    def anti_role(self):
        return leader if self.follower else follower

    @property
    def leader(self):
        return self.role == leader

    @property
    def follower(self):
        return self.role == follower

    @property
    def received_version(self):
        """received version equals to other node upload version.
        """
        return self.get('upload_version')

    @property
    def request_version(self):
        """request version equals to other node download version.
        """
        return self.get('download_version')

    @property
    def nick_name(self):
        return self.get(nick_name)

    def __init__(
        self,
        store: Store,
        pg: ProcessGroup,
        dist_props: DistributedProperties,
        fed_props: FederatedProperties,
    ):
        self.store = store
        self.pg = pg
        self.dist_props = dist_props
        self.fed_props = fed_props

        set_store_value(store=self.store,
                        key=self._i_key,
                        value={
                            openfed_status: zombie,
                            openfed_task_info: TaskInfo(),
                            nick_name: self.fed_props.nick_name,
                            "upload_version": -1,
                            "download_version": -1,
                        })

        self._i_backup_info = get_store_value(self.store, self._i_key)
        self._u_backup_info = get_store_value(self.store, self._u_key)
        
        self.read_successfully = True

    def set_upload_version(self, version):
        self.set("upload_version", version)

    def set_download_version(self, version):
        self.set("download_version", version)

    def upload(self, data, version: Optional[Any] = None):
        if version is not None:
            self.set_upload_version(version)
        self.transfer(False, data)

    def download(self, version: Optional[Any] = None) -> Any:
        if version is not None:
            self.set_download_version(version)
        return self.transfer(True)

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
        if self.is_offline: raise DeviceOffline(self)

        def _state():
            return self.is_pulling if to else self.is_pushing

        if self.follower:
            if to:
                self.pushing()
            else:
                self.pulling()

            tic = time.time()
            while not _state():
                if self.is_offline: raise DeviceOffline(self)
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
            "Pipe is only designed for point to point communication."

        rank = follower_rank if self.leader else leader_rank
        rank = distributed_c10d._get_global_rank(self.pg, rank) \
            if distributed_c10d.get_world_size() > 2 else rank

        gather_object(data, None, dst=rank, group=self.pg)

    def pull(self) -> Any:
        assert distributed_c10d._get_group_size(self.pg) == 2,\
            "Pipe is only designed for point to point communication."

        world_size = distributed_c10d.get_world_size()

        received = [None for _ in range(world_size)]

        rank = leader_rank if self.leader else follower_rank
        other_rank = follower_rank if self.leader else leader_rank

        rank = distributed_c10d._get_global_rank(
            self.pg, rank) if world_size > 2 else rank
        other_rank = distributed_c10d._get_global_rank(
            self.pg, other_rank) if world_size > 2 else other_rank

        gather_object(None, received, dst=rank, group=self.pg)

        data = [r for r in received if r is not None][0]
        assert data

        return data

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
        head = ['nick name', 'received version', 'request version', 'status']
        data = [
            self.nick_name, self.received_version, self.request_version,
            self._get_state()
        ]
        description = tablist(head, data, force_in_one_row=True)

        other_description = "dist props: \n" + str(self.dist_props) + \
            "fed props: \n " + str(self.fed_props)
        return openfed_class_fmt.format(class_name=self.__class__.__name__,
                                        description=description + "\n" +
                                        other_description)
