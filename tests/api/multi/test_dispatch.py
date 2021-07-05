from openfed.api.multi.dispatch import Dispatch


def test_dispatch():
    Dispatch(1, samples=1)
    Dispatch(1, sample_ratio=1)