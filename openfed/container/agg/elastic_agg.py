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
from openfed.common import TaskInfo
from .agg import Agg


class ElasticAgg(Agg):
    """A data-award aggregation method.

    This aggregator must be paired with `elastic pipe`.
    """

    def __init__(self, params,
                 other_keys: Union[str, List[str]] = None,
                 quantile  : float                 = 0.5,
                 legacy    : bool                  = True):

        other_keys = [] if other_keys is None else convert_to_list(other_keys)

        if not (0 < quantile < 1.0):
            raise ValueError("quantile must be between 0 and 1")

        info_keys: List[str] = ['instances']
        pipe_keys: List[str] = [
            "step", "received_params", "param", "importance"]

        for k in other_keys:
            if k in pipe_keys:
                raise ValueError(f"Duplicate key: {k}")

        pipe_keys.extend(other_keys)

        defaults = dict(quantile=quantile,)
        super().__init__(
            params,
            defaults,
            info_keys = info_keys,
            pipe_keys = pipe_keys,
            legacy    = legacy)

    def merge(self, 
            p            : Tensor,
            r_p          : Dict[str, Tensor],
            received_info: TaskInfo,
            group        : Dict) -> Any     : 
        instances = received_info.instances
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        for key in group['pipe_keys']:
            if key in r_p:
                state[key] = r_p[key] if key not in state else (
                    state[key] * step + r_p[key] * instances) / (step + instances)
        state['step'] += instances

    def stack(self, 
            p            : Tensor,
            r_p          : Dict[str, Tensor],
            received_info: TaskInfo,
            **unused) -> Any:
        state = self.state[p]
        if 'received_params' not in state:
            state['received_params'] = []

        r_p["instances"] = received_info.instances
        state['received_params'].append(r_p)

    def _merge_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]

        for key in group['pipe_keys']:
            if key in state:
                new_p = state[key]
                if key == "param":
                    if p.requires_grad:
                        if p.grad is not None:
                            p.grad.copy_(self._elastic_update(
                                p-new_p, state['importance'], group['quantile']))
                        else:
                            p.grad = self._elastic_update(
                                p-new_p, state['importance'], group['quantile'])
                    else:
                        p.copy_(new_p)
                else:
                    ...

    def _stack_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]

        def aggregate(dl, k, t) -> Tensor:
            l: List[Tensor] = []
            for data in dl:
                a, b = data[k], data['instances']
                w  = b / t
                p  = a * w
                l.append(p)
            return torch.stack(l, dim=0).sum(dim=0, keepdim=False)

        total_instances = 0
        for data in state["received_params"]:
            instances = data["instances"]
            total_instances += instances

        for key in group['pipe_keys']:
            if key in state["received_params"][0]:
                new_p = aggregate(
                    state["received_params"], key, total_instances)
                if key == "param":
                    if p.requires_grad:
                        new_imp = aggregate(
                            state["received_params"], "importance", total_instances)
                        grad = self._elastic_update(
                            p-new_p, new_imp, group["quantile"])
                        if p.grad is None:
                            p.grad = grad
                        else:
                            p.grad.copy_(grad)
                    else:
                        p.copy_(new_p)
                else:
                    state[key] = new_p

    def _elastic_update(self, 
        grad      : Tensor,
        importance: Tensor,
        quantile  : float): 
        norm_importance = importance / (importance.max() + 1e-13)
        weight          = 1 + quantile - norm_importance

        return grad * weight
