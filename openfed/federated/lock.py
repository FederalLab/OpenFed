from collections import OrderedDict
from threading import Lock
from typing import Dict

from openfed.common.logging import logger


class Maintainer():
    # define here to avoid circular import error.
    ...


_maintainer_lock_dict: Dict[Maintainer, Lock] = OrderedDict()

openfed_lock = Lock()


def add_maintainer_lock(maintainer: Maintainer, lock: Lock):
    _maintainer_lock_dict[maintainer] = lock


def del_maintainer_lock(maintainer: Maintainer):
    if maintainer in _maintainer_lock_dict:
        del _maintainer_lock_dict[maintainer]
    else:
        logger.debug("Maintainer lock is already deleted.")


def acquire_all():
    for maintainer_lock in _maintainer_lock_dict.values():
        maintainer_lock.acquire()
    openfed_lock.acquire()


def release_all():
    for maintainer_lock in _maintainer_lock_dict.values():
        maintainer_lock.release()
    openfed_lock.release()
