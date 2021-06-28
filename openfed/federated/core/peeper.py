from openfed.common.thread import _thread_pool
from openfed.utils import tablist

from ..space.world import _world_list
from ..utils.lock import _maintainer_lock_dict
from ..utils.register import _country


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
