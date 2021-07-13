from .optim import Optimizer
from .pena import Penalizer
from openfed.utils import glue


class Pipe(Optimizer, Penalizer):
    """Glue Optimizer and Penalizer as a single class.
    """
    ...


def build_pipe(optimizer: Optimizer, penalizer: Penalizer) -> Pipe:
    return glue(optimizer, penalizer)
