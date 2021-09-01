from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import torch.distributed.distributed_c10d as distributed_c10d
from openfed.common import Address
from openfed.utils import openfed_class_fmt, tablist

from .const import default_pg_timeout, follower_rank, leader_rank


class DistributedProperties(object):
    # Use to save the default distributed probabilities
    _default_WORLD = distributed_c10d.group.WORLD
    _default_pg_map = distributed_c10d._pg_map
    _default_pg_names = distributed_c10d._pg_names
    _default_pg_group_ranks = distributed_c10d._pg_group_ranks
    _default_pg_init_method = distributed_c10d._default_pg_init_method
    _default_group_count = distributed_c10d._group_count

    # Use to save current distributed probabilities
    _WORLD: Optional[distributed_c10d.ProcessGroup] = None
    _pg_map: Dict[distributed_c10d.ProcessGroup,
                  Tuple[str, Optional[distributed_c10d.Store]]] = {}
    _pg_names: Dict[distributed_c10d.ProcessGroup, str]
    _pg_group_ranks: Dict[distributed_c10d.ProcessGroup, Dict[int, int]]
    _pg_init_method: Any
    _group_count: int

    def __init__(self, lock: Optional[Lock] = None):
        self._WORLD = None
        self._pg_map = {}
        self._pg_names = {}
        self._pg_group_ranks = {}
        self._pg_init_method = None
        self._group_count = 0

        self.lock = lock or Lock()  # type: Lock

    def __enter__(self):
        self.lock.acquire()

        # save default value and load individual value
        DistributedProperties._default_WORLD = distributed_c10d.group.WORLD
        distributed_c10d.group.WORLD = self._WORLD
        distributed_c10d.GroupMember.WORLD = self._WORLD

        DistributedProperties._default_pg_map = distributed_c10d._pg_map
        distributed_c10d._pg_map = self._pg_map

        DistributedProperties._default_pg_names = distributed_c10d._pg_names
        distributed_c10d._pg_names = self._pg_names

        DistributedProperties._default_pg_group_ranks = distributed_c10d._pg_group_ranks
        distributed_c10d._pg_group_ranks = self._pg_group_ranks

        DistributedProperties._default_pg_init_method = distributed_c10d._default_pg_init_method
        distributed_c10d._default_pg_init_method = self._pg_init_method

        DistributedProperties._default_group_count = distributed_c10d._group_count
        distributed_c10d._group_count = self._group_count

    def __exit__(self, exc_type, exc_value, trace):
        # save individual value and load default value
        self._WORLD = distributed_c10d.group.WORLD
        distributed_c10d.group.WORLD = DistributedProperties._default_WORLD
        distributed_c10d.GroupMember.WORLD = DistributedProperties._default_WORLD

        self._pg_map = distributed_c10d._pg_map
        distributed_c10d._pg_map = DistributedProperties._default_pg_map

        self._pg_names = distributed_c10d._pg_names
        distributed_c10d._pg_names = DistributedProperties._default_pg_names

        self._pg_group_ranks = distributed_c10d._pg_group_ranks
        distributed_c10d._pg_group_ranks = DistributedProperties._default_pg_group_ranks

        self._pg_init_method = distributed_c10d._default_pg_init_method
        distributed_c10d._default_pg_init_method = DistributedProperties._default_pg_init_method

        self._group_count = distributed_c10d._group_count
        distributed_c10d._group_count = DistributedProperties._default_group_count

        self.lock.release()

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name=self.__class__.__name__,
            description=f"GroupCount: {self._group_count}")


def build_point2point_group(
        rank: int = 0) -> List[distributed_c10d.ProcessGroup]:
    if distributed_c10d.get_world_size() == 2:
        return [distributed_c10d._get_default_group()]

    assert 0 <= rank < distributed_c10d.get_world_size()

    pg_list = []
    for other in range(distributed_c10d.get_world_size()):
        if other != rank:
            ranks = [other, rank] if follower_rank == 0 else [rank, other]
            pg = distributed_c10d.new_group(ranks=ranks)
            if pg is not distributed_c10d.GroupMember.NON_GROUP_MEMBER:
                pg_list.append(pg)
    return pg_list


def joint_federated_group(
    backend,
    init_method=None,
    world_size=-1,
    rank=-1,
):
    # build a store
    rendezvous_iterator = distributed_c10d.rendezvous(
        init_method, rank, world_size, timeout=default_pg_timeout)
    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(distributed_c10d.default_pg_timeout)

    distributed_c10d.init_process_group(backend,
                                        world_size=world_size,
                                        rank=rank,
                                        store=store)
    # rank is always set to 0 for that we want to build a
    # point2point connection between the master and each nodes.
    # If the address is a point2point one, we should use the leader rank.
    # If the address is a shared multi-node one, we take the rank 0 as the leader rank.
    # And the re-arranged rank will be set to the ideal rank order in function call.
    sub_pg_list = build_point2point_group(
        leader_rank if distributed_c10d.get_world_size() == 2 else 0)

    return sub_pg_list


class FederatedProperties(object):
    role: str
    nick_name: str
    address: Address
    mtt: int

    def __init__(self,
                 role: str,
                 nick_name: str,
                 address: Address,
                 mtt: int = 5):
        self.role = role
        self.nick_name = nick_name
        self.address = address
        self.mtt = mtt

    def __repr__(self):
        head = ['role', 'nick_name', 'mtt']
        data = [self.role, self.nick_name, self.mtt]
        description = tablist(head, data, force_in_one_row=True)
        other_description = str(self.address)
        return openfed_class_fmt.format(class_name=self.__class__.__name__,
                                        description=description + "\n" +
                                        other_description)
