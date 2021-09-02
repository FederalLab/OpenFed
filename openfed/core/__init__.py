# Copyright (c) FederalLab. All rights reserved.
from .const import *
from .federated import *
from .pipe import *

del pipe
del const
del federated
import warnings


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
