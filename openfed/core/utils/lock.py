# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from collections import OrderedDict
from threading import Lock
from typing import Dict

from openfed.common import logger


class Maintainer():
    # define here to avoid circular import error.
    ...


_maintainer_lock_dict: Dict[Maintainer, Lock] = OrderedDict()

openfed_lock = Lock()


def add_mt_lock(maintainer: Maintainer, lock: Lock):
    _maintainer_lock_dict[maintainer] = lock


def del_maintainer_lock(maintainer: Maintainer):
    if maintainer in _maintainer_lock_dict:
        del _maintainer_lock_dict[maintainer]
    else:
        logger.debug("Maintainer lock is already deleted.")


def acquire_all():
    for mt_lock in _maintainer_lock_dict.values():
        mt_lock.acquire()
    openfed_lock.acquire()


def release_all():
    for mt_lock in _maintainer_lock_dict.values():
        mt_lock.release()
    openfed_lock.release()
