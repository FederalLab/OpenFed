from openfed.common.constants import *


def test_constants():
    assert DEFAULT_PG_TIMEOUT.seconds > 0
    assert DEFAULT_PG_LONG_TIMEOUT.seconds > 0
    assert DEFAULT_PG_SHORT_TIMEOUT.seconds > 0
