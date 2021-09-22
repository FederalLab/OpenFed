# Copyright (c) FederalLab. All rights reserved.
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import torch.distributed.distributed_c10d as distributed_c10d

from openfed.common import Address
from openfed.utils import openfed_class_fmt, tablist


class DistributedProperties(object):
    r'''Keeps all distributed properties in this class, so that we can build
    multi-process groups.

    Args:
        lock: A lock used to protect distributed properties. Default: ``None``

    Examples::

        >>> dist_props = DistributedProperties()
        >>> dist_props
        <OpenFed> DistributedProperties
        GroupCount: 0

        >>> with dist_props:
        ...     ...
        ...
        Ellipsis
    '''
    # Use to save the default distributed probabilities
    _default_WORLD = distributed_c10d.group.WORLD
    _default_pg_map = distributed_c10d._pg_map
    _default_pg_names = distributed_c10d._pg_names
    _default_pg_group_ranks = distributed_c10d._pg_group_ranks
    _default_pg_init_method = distributed_c10d._default_pg_init_method
    _default_group_count = distributed_c10d._group_count

    # Use to save current distributed probabilities
    _WORLD: Optional[Any] = None
    _pg_map: Dict[Any, Tuple[str, Optional[Any]]] = {}
    _pg_names: Dict[Any, str]
    _pg_group_ranks: Dict[Any, Dict[int, int]]
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

        DistributedProperties._default_pg_group_ranks = \
            distributed_c10d._pg_group_ranks
        distributed_c10d._pg_group_ranks = self._pg_group_ranks

        DistributedProperties._default_pg_init_method = \
            distributed_c10d._default_pg_init_method
        distributed_c10d._default_pg_init_method = self._pg_init_method

        DistributedProperties._default_group_count = \
            distributed_c10d._group_count
        distributed_c10d._group_count = self._group_count

    def __exit__(self, exc_type, exc_value, trace):
        # save individual value and load default value
        self._WORLD = distributed_c10d.group.WORLD
        distributed_c10d.group.WORLD = DistributedProperties._default_WORLD
        distributed_c10d.GroupMember.WORLD = \
            DistributedProperties._default_WORLD

        self._pg_map = distributed_c10d._pg_map
        distributed_c10d._pg_map = DistributedProperties._default_pg_map

        self._pg_names = distributed_c10d._pg_names
        distributed_c10d._pg_names = DistributedProperties._default_pg_names

        self._pg_group_ranks = distributed_c10d._pg_group_ranks
        distributed_c10d._pg_group_ranks = \
            DistributedProperties._default_pg_group_ranks

        self._pg_init_method = distributed_c10d._default_pg_init_method
        distributed_c10d._default_pg_init_method = \
            DistributedProperties._default_pg_init_method

        self._group_count = distributed_c10d._group_count
        distributed_c10d._group_count = \
            DistributedProperties._default_group_count

        self.lock.release()

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name=self.__class__.__name__,
            description=f'GroupCount: {self._group_count}')


class FederatedProperties(object):
    r'''Keeps all federated properties in this class, so that we can build
    multi-process groups.

    Args:
        role: The role played.
        nick_name: The name of node.
        address: The address to connect this federated group.

    Examples::

        >>> fed_props = FederatedProperties('aggregator',
        >>>     'openfed_node', default_tcp_address)
        >>> fed_props
        <OpenFed> FederatedProperties
        +--------+--------------+
        |  role  |  nick_name   |
        +--------+--------------+
        | aggregator | openfed_node |
        +--------+--------------+
        <OpenFed> Address
        +---------+---------------------+------------+------+
        | backend |     init_method     | world_size | rank |
        +---------+---------------------+------------+------+
        |   gloo  | tcp://localhost:... |     2      |  -1  |
        +---------+---------------------+------------+------+
    '''
    role: str
    nick_name: str
    address: Address

    def __init__(self, role: str, nick_name: str, address: Address):
        self.role = role
        self.nick_name = nick_name
        self.address = address

    def __repr__(self):
        head = ['role', 'nick_name']
        data = [self.role, self.nick_name]
        description = tablist(head, data, force_in_one_row=True)
        other_description = str(self.address)
        return openfed_class_fmt.format(
            class_name=self.__class__.__name__,
            description=description + '\n' + other_description)
