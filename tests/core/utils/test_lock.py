from openfed.core.utils.lock import *


def test_lock():
    acquire_all()

    release_all()
