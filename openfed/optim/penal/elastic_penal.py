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


from typing import List, Callable, Union

import torch
from openfed.core import follower, leader
from openfed.common import Package
from openfed.utils import convert_to_list
from typing_extensions import final
import torch.nn.functional as F

from .penal import Penalizer

class ElasticPenalizer(Penalizer):
    r"""Elastic Penalizer is used for collecting some training statics 
    of client's data. Actually, it can be glued with other penalizers 
    to create new Pipe.

    .. note::
        Not any two penalizer can be glued, you have to make sure that
        the methods are not conflict. ElasticPenalizer only use the `acg_step`, 
        which make it couple well with other penalizers.
    """

    def __init__(self,
                 role: str,
                 momentum: float = 0.9,
                 pack_set: List[str] = None,
                 unpack_set: List[str] = None,
                 max_acg_step: int = -1):
        """
        Args:
            role: The role played.
            momentum: The momentum to accumulate importance weight.
            pack_set: ...
            unpack_set: ...
        """
        pack_set = convert_to_list(pack_set) or []
        pack_set.append('importance')

        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        # Do not record momentum to defaults.
        # It will make conflict with the momentum in other optimizer, 
        # such as SGD.
        self.momentum = momentum

        super().__init__(role, pack_set, unpack_set, max_acg_step)

    def acg(self, model, dataloader):
        """Accumulate gradients for elastic aggregation.
        Args:
            model: The model used to test.
            dataloader: The dataloader used to iterate over.

        The dataloader should return with [data, target] tuple.
        This is often used for classification task. 
        """
        model.train()
        device = next(model.parameters()).device

        for i, data in enumerate(dataloader):
            input, _ = data
            input = input.to(device)

            model.zero_grad()
            output = model(input)
            F.mse_loss(output, torch.zeros_like(output)).backward()
            self._acg_step()

            if self.max_acg_step > 0 and i > self.max_acg_step:
                break

    def _acg_step(self):
        """Performs a single accumulate gradient step.
        """
        for group in self.param_groups:
            momentum = self.momentum
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.abs()
                state = self.state[p]
                if 'importance' not in state:
                    state["importance"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                state["importance"].mul_(momentum).add_(grad, alpha=1-momentum)

