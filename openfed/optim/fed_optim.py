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


class FedOptim(Optimizer, Penalizer):
    """Glue Optimizer and Penalizer as a single class.
    """
    def __new__(cls):
        raise RuntimeError("Call `build_fed_optim` to create a fed_optim.")


def build_fed_optim(optimizer: Optimizer, penalizer: Penalizer = None) -> FedOptim:
    """Glue optimizer and penalizer as a single class, named `FedOptim`.
    Args:
        optimizer: The torch optimizer.
        penalizer: The federated penalizer.
    
    .. note::
        Penalizer has no influence on optimizer's behavior. You can just
        write such code: `optim=build_fed_optim(optim)`. 
    """
    # As for Penalizer(), it do noting, so the role is not related.
    penalizer = penalizer or Penalizer()

    def step(func_a, func_b):
        # Create a decorator that glue func_a and func_b.
        def _step(*args, **kwargs):
            # If the output is dictionary, we will return output_a.update(output_b)
            # Otherwise, we only return `output_a or output_b`
            output_a = func_a(*args, **kwargs)
            output_b = func_b(*args, **kwargs)
            if isinstance(output_a, dict) and isinstance(output_b, dict):
                output_a.update(output_b)
                return output_a
            return output_a or output_b
        return _step
    OptimizerT = type(optimizer)
    PenalizerT = type(penalizer)
    # Penalizer step should be first steped.
    extra_func = dict(step=step(
        func_a=getattr(PenalizerT, 'step'), 
        func_b=getattr(OptimizerT, 'step')))
    return glue(optimizer, penalizer, extra_func) # type: ignore