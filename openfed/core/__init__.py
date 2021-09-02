# Copyright (c) FederalLab. All rights reserved.
import warnings
from threading import Lock

from .const import *
from .federated import *
from .pipe import *

openfed_lock = Lock()


def init_federated_group(fed_props: FederatedProperties) -> List[Pipe]:
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
        for sub_pg in sub_pg_list:
            store = distributed_c10d._pg_map[sub_pg][1]
            pipe = Pipe(
                store,  # type: ignore
                pg=sub_pg,
                dist_props=dist_props,
                fed_props=fed_props)
            pipe_list.append(pipe)

    return pipe_list


__all__ = [
    'leader',
    'follower',
    'is_leader',
    'is_follower',
    'push',
    'pull',
    'zombie',
    'offline',
    'openfed_identity',
    'openfed_status',
    'openfed_meta',
    'nick_name',
    'leader_rank',
    'follower_rank',
    'default_pg_timeout',
    'DistributedProperties',
    'build_point2point_group',
    'joint_federated_group',
    'FederatedProperties',
    'set_store_value',
    'get_store_value',
    'Pipe',
    'init_federated_group',
]
