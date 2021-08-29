from .const import *
from .federated import *
from .pipe import *
import warnings

del pipe
del const
del federated


def init_federated_group(federated_group_properties):
    pipe_list = []
    finished = []

    address_list = federated_group_properties.address_list
    distributed_properties_list = [
        DistributedProperties(openfed_lock) for _ in range(len(address_list))
    ]
    tt = 0

    while tt < federated_group_properties.mtt:
        for address, dist_prop in zip(address_list,
                                      distributed_properties_list):
            if address not in finished:
                with dist_prop:
                    try:
                        sub_pg_list = joint_federated_group(*address)
                    except TimeoutError as e:
                        warnings.warn(str(e))
                        continue
                    # build pipe
                    for sub_pg in sub_pg_list:
                        store = distributed_c10d._pg_map[sub_pg][1]
                        pipe = Pipe(
                            store,  # type: ignore
                            pg=sub_pg,
                            distributed_properties=dist_prop,
                            federated_group_properties=
                            federated_group_properties)
                        pipe_list.append(pipe)
                    finished.append(address)
        tt += 1
        if len(finished) == len(address_list):
            break
    return pipe_list
