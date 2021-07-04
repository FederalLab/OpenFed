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


from typing import Any, Dict, List, Union

import torch
from openfed.utils import convert_to_list
from torch import Tensor

from .agg import Agg


class AverageAgg(Agg):
    """average all received data directly.
    """

    def __init__(self,
                 params,
                 other_keys: Union[str, List[str]] = None,
                 legacy    : bool                  = True):
        """
        Args:
            other_keys: any keys you want to track.
        """
        other_keys = [] if other_keys is None else convert_to_list(other_keys)

        info_keys: List[str] = []
        pipe_keys: List[str] = ["step", "received_params", "param"]

        for k in other_keys:
            if k in pipe_keys:
                raise ValueError(f"Duplicate key: {k}")

        pipe_keys.extend(other_keys)
        defaults = dict()
        super().__init__(
            params,
            defaults,
            info_keys = info_keys,
            pipe_keys = pipe_keys,
            legacy    = legacy)

    def merge(self, p: Tensor, r_p: Dict[str, Tensor], received_info: Dict, group: Dict) -> Any:
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        for key in group['pipe_keys']:
            if key in r_p:
                state[key] = r_p[key] if key not in state else (
                    state[key] * step + r_p[key]) / (step + 1)
        state['step'] += 1

    def stack(self, p: torch.Tensor, r_p: Dict[str, Tensor], **unused) -> Any:
        state = self.state[p]

        if 'received_params' not in state:
            state['received_params'] = []
        state['received_params'].append(r_p)

    def _merge_aggregate(self, p: torch.Tensor, group: Dict):
        state = self.state[p]
        pipe_keys = group['pipe_keys']

        for key in pipe_keys:
            if key in state:
                new_p = state[key]
                if key == "param":
                    if p.requires_grad:
                        # NOTE: grad = p - new_p
                        p.grad.copy_(p - new_p)
                    else:
                        p.copy_(new_p)
                else:
                    ...

    def _stack_aggregate(self, p: torch.Tensor, group: Dict):
        state = self.state[p]

        def agg(dl, k):
            return torch.stack([data[k] for data in dl], dim=0).mean(dim=0, keepdims=False)

        pipe_keys = group['pipe_keys']
        for key in pipe_keys:
            if key in state['received_params'][0]:
                new_p = agg(state["received_params"], key)
                if key == "param":
                    if p.requires_grad:
                        if p.grad is None:
                            p.grad = p-new_p
                        else:
                            p.grad.copy_(p-new_p)
                    else:
                        p.copy_(new_p)
                else:
                    state[key] = new_p
