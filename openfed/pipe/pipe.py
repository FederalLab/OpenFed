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
 

from openfed.common import glue
from torch.optim import Optimizer

from .penal import Penalizer


class Pipe(Optimizer, Penalizer):
    """Glue Optimizer and Penalizer as a single class.
    """
    def __new__(cls):
        raise RuntimeError("Call `build_pipe` to create a pipe.")


def build_pipe(optimizer: Optimizer, penalizer: Penalizer = None) -> Pipe:
    """Glue optimizer and penalizer as a single class, named `Pipe`.
    Args:
        optimizer: The torch optimizer.
        penalizer: The federated penalizer.
    
    .. note::
        Penalizer has no influence on optimizer's behavior. You can just
        write such code: `optim=build_pipe(optim)`. 
    """
    penalizer = Penalizer() if penalizer is None else penalizer
    extra_func = dict(step=None)
    return glue(optimizer, penalizer, extra_func) # type: ignore
