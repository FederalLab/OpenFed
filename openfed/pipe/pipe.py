from .optim import Optimizer
from .pena import Penalizer
from openfed.utils import glue


class Pipe(Optimizer, Penalizer):
    """Glue Optimizer and Penalizer as a single class.
    """
    def __new__(cls):
        raise RuntimeError("Call `build_pipe` to create a pipe.")


def build_pipe(optimizer: Optimizer, penalizer: Penalizer = None) -> Pipe:
    penalizer = Penalizer() if penalizer is None else penalizer
    return glue(optimizer, penalizer)
