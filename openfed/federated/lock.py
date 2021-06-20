from collections import OrderedDict
from threading import Lock
from typing import Dict

import openfed
from openfed.common.logging import logger


class Maintainer():
    # define here to avoid circular import error.
    ...


_maintainer_lock_dict: Dict[Maintainer, Lock] = OrderedDict()

openfed_lock = Lock()


def add_maintainer_lock(maintainer, lock):
    _maintainer_lock_dict[maintainer] = lock


def del_maintainer_lock(maintainer):
    if maintainer in _maintainer_lock_dict:
        del _maintainer_lock_dict[maintainer]
    else:
        if openfed.DEBUG.is_debug:
            logger.error("Maintianer lock is already deleted.")


def acquire_all():
    for maintainer_lock in _maintainer_lock_dict.values():
        maintainer_lock.acquire()
    openfed_lock.acquire()


def release_all():
    for maintainer_lock in _maintainer_lock_dict.values():
        maintainer_lock.release()
    openfed_lock.release()
