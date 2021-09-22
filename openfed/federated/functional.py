# Copyright (c) FederalLab. All rights reserved.
import warnings
from datetime import timedelta
from threading import Lock
from typing import Any, List, Union

import torch.distributed.distributed_c10d as distributed_c10d
from torch._C._distributed_c10d import PrefixStore
from torch.distributed.constants import default_pg_timeout
from torch.distributed.rendezvous import rendezvous

from .const import aggregator_rank, collaborator_rank
from .pipe import Pipe
from .props import DistributedProperties, FederatedProperties

openfed_default_pg_timeout = timedelta(seconds=100)


def build_point2point_group(rank: int = 0) -> List[Union[str, Any]]:
    r'''Builds process groups between two ranks.

    Args:
        rank: The rank of the target process group. Default: ``0``

    Returns:
        List contains :class:`ProcessGroup`.
    '''
    if distributed_c10d.get_world_size() == 2:
        return [distributed_c10d._get_default_group()]

    assert 0 <= rank < distributed_c10d.get_world_size()

    pg_list = []
    for other in range(distributed_c10d.get_world_size()):
        if other != rank:
            ranks = [other, rank] if collaborator_rank == 0 else [rank, other]
            pg = distributed_c10d.new_group(ranks=ranks)
            if pg is not distributed_c10d.GroupMember.NON_GROUP_MEMBER:
                pg_list.append([f'from_{ranks[0]}_to_{ranks[1]}', pg])
    return pg_list


def joint_federated_group(backend,
                          init_method=None,
                          world_size=-1,
                          rank=-1) -> List:
    r'''Joints federated group. We will build a store manually used for meta
    exchange.

    .. warning ::
        ``env://`` is not allowed to be used to avoid conflicts with
        distributed learning in `PyTorch`.

    Args:
        backend: The backend to used. Depending on build-time configurations,
            valid values include ``mpi``, ``gloo`` and  ``nccl``, which depends
            on the `PyTorch` you installed. This field should be given as a
            lowercase string (e.g., ``gloo``), which can also be accessed via
            :class:`Backend` attributes (e.g., ``Backend.GLOO``). ``nccl`` is
            not recommended currently, for that we will always move the tensor
            to `cpu` first before sending to other nodes to avoiding device
            disalignment between `cpu` and `gpu`. Thus, ``nccl`` will not speed
            up the communication phase in `OpenFed`. Default: ``'gloo'``.
        init_method: URL specifying how to initialize the federated group. Such
            as: ``tcp://localhost:1994``, ``file:///tmp/sharefile``. If you use
            ``file://``, make sure the file is not existing.
            Default: ``'tcp://localhost:1994'``
        world_size: Number of nodes in federated group. Default: ``2``
        rank: Rank of current node (it should be a number between 0 and
            ``world_size``-1). If `-1` is provided, rank will be specified
            during runtime. Default: -1

    Returns:
        List contains :class:`ProcessGroup`.
    '''
    # build a store
    rendezvous_iterator = rendezvous(
        init_method, rank, world_size, timeout=openfed_default_pg_timeout)
    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(default_pg_timeout)

    distributed_c10d.init_process_group(
        backend, world_size=world_size, rank=rank, store=store)
    # rank is always set to 0 for that we want to build a
    # point2point connection between the master and each nodes.
    # If the address is a point2point one, we should use the aggregator rank.
    # If the address is a shared multi-node one, we take the rank 0 as
    # the aggregator rank.
    # And the re-arranged rank will be set to the ideal rank order
    # in function call.
    sub_pg_list = build_point2point_group(
        aggregator_rank if distributed_c10d.get_world_size() == 2 else 0)

    return sub_pg_list


openfed_lock = Lock()


def init_federated_group(fed_props: FederatedProperties) -> List[Pipe]:
    r"""Initializes federated group.

    Args:
        fed_props: The federated properties to build.

    Returns:
        List contains :class:`Pipe`.
    """
    dist_props = DistributedProperties(openfed_lock)
    pipe_list = []
    with dist_props:
        address = fed_props.address
        try:
            sub_pg_list = joint_federated_group(
                backend=address.backend,
                init_method=address.init_method,
                world_size=address.world_size,
                rank=address.rank,
            )
        except RuntimeError as e:
            warnings.warn(str(e))
            return []
        # build pipe
        for (prefix, sub_pg) in sub_pg_list:
            store = distributed_c10d._pg_map[sub_pg][1]
            assert store
            prefix_store = PrefixStore(prefix, store)
            pipe = Pipe(
                prefix_store,
                pg=sub_pg,
                dist_props=dist_props,
                fed_props=fed_props)
            pipe_list.append(pipe)

    return pipe_list
