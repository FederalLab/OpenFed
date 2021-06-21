from openfed.federated.world import _world_list
from openfed.federated.register import _country
from openfed.federated.lock import _maintainer_lock_dict
from openfed.common.thread import _thread_pool


class Peeper(object):
    """
    Collect the state of whole openfed.
    """

    @classmethod
    def openfed_digest(cls):
        return (
            f"OpenFed Digest\n"
            f"World: {len(_world_list)}\n"
            f"Country: {len(_country)}\n"
            f"Maintainer: {len(_maintainer_lock_dict)}\n"
            f"Thread: {len(_thread_pool)}\n"
        )
