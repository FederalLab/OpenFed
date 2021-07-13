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


from typing import List

import torch
from openfed.utils import convert_to_list

from .penal import Penalizer


class ElasticPenalizer(Penalizer):
    r"""Paired with ElasticAgg.

    Example:
        >>> elastic_pipe = ElasticPipe(net.parameters(), momentum=0.9)
        >>> while:
        >>>     elastic_pipe.zero_grad()
        >>>     MSE(net(input), zeros).backward()
        >>>     elastic_pipe.step()
        >>> elastic_pipe.clear_state()

    """

    def __init__(self,
                 ft: bool,
                 momentum: float = 0.9,
                 pack_key_list: List[str] = None,
                 unpack_key_list: List[str] = None):
        pack_key_list = convert_to_list(pack_key_list)
        unpack_key_list = convert_to_list(unpack_key_list)
        if pack_key_list is None:
            pack_key_list = ['importance']
        else:
            pack_key_list.append('importance')

        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        self.momentum = momentum

        super().__init__(ft, pack_key_list, unpack_key_list)

    def acg_step(self):
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
