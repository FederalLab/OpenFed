from openfed.api.multi.dispatch import Dispatch


def test_dispatch():
    Dispatch(total_parts=1, samples=1)
    Dispatch(total_parts=1, sample_ratio=1)