from openfed.common import glue
from torch.optim import Optimizer

from .penal import Penalizer


class Pipe(Optimizer, Penalizer):
    """Glue Optimizer and Penalizer as a single class.
    """
    def __new__(cls):
        raise RuntimeError("Call `build_pipe` to create a pipe.")


def build_pipe(optimizer: Optimizer, penalizer: Penalizer = None) -> Pipe:
    penalizer = Penalizer() if penalizer is None else penalizer
    extra_func = dict(step=None)
    return glue(optimizer, penalizer, extra_func) # type: ignore
