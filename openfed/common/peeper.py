from openfed.common.thread import _thread_pool
from openfed.federated.space.world import _world_list
from openfed.federated.utils.lock import _maintainer_lock_dict
from openfed.federated.utils.register import _country
from openfed.utils.table import tablist


class Peeper(object):
    """
    Collect the state of whole openfed.
    """

    @classmethod
    def openfed_digest(cls):
        return tablist(
            head=['World', 'Country', 'Maintainer', 'Thread'],
            data=[len(_world_list), len(_country), len(
                _maintainer_lock_dict), len(_thread_pool)],
            force_in_one_row=True
        )
