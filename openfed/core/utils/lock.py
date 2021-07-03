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


from threading import Lock

from openfed.common.base import peeper

peeper.add_to_peeper('mt_locks', dict())
openfed_lock = Lock()


def add_mt_lock(maintainer, lock: Lock):
    mt_locks = peeper.get_from_peeper('mt_locks')
    mt_locks[maintainer] = lock


def del_maintainer_lock(maintainer):
    mt_locks = peeper.get_from_peeper('mt_locks')
    if maintainer in mt_locks:
        del mt_locks[maintainer]


def acquire_all():
    mt_locks = peeper.get_from_peeper('mt_locks')
    for mt_lock in mt_locks.values():
        mt_lock.acquire()
    openfed_lock.acquire()


def release_all():
    mt_locks = peeper.get_from_peeper('mt_locks')
    for mt_lock in mt_locks.values():
        mt_lock.release()
    openfed_lock.release()
