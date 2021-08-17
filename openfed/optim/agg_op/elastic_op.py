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

from typing import Any, Dict, List

import torch
from openfed.common import TaskInfo
from openfed.utils import convert_to_list
from torch import Tensor

from .agg_op import AggOp

class ElasticOp(AggOp):
    """Elastic Aggregation: Aggregate received tensor in a data-aware way.

    .. warn::
        ElasticAgg must be coupled with `ElasticPenalizer`.
    """

    def __init__(self,
                 params,
                 quantile: float = 0.5,
                 stack: bool = False,
                 other_keys: List[str] = None):
        """
        Args:
            quantile: The quantile point that magnify or suppress the parameter's gradient.
        """

        other_keys = convert_to_list(other_keys) or []

        if not (0 < quantile < 1.0):
            raise ValueError("quantile must be between 0 and 1")

        info_keys = ['instances']
        pipe_keys = list(
            set(["step", "received_params", "param", "importance"] + other_keys))

        defaults = dict(
            quantile=quantile,
            info_keys=info_keys,
            pipe_keys=pipe_keys,
            stack=stack
        )
        super().__init__(params, defaults)

    def merge(self,
              p: Tensor,
              r_p: Dict[str, Tensor],
              r_info: TaskInfo,
              group: Dict) -> Any:
        """Merge received tensor to elastic aggregator buffer.
        """
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        instances = r_info.instances  # type: ignore
        for key in group['pipe_keys']:
            if key in r_p:
                # Merge the received tensor to previous stored one
                # Here, a weighted average is applied.
                state[key] = r_p[key] if key not in state else (
                    state[key] * step + r_p[key] * instances) / (step + instances)
        state['step'] += instances

    def stack(self,
              p: Tensor,
              r_p: Dict[str, Tensor],
              r_info: TaskInfo,
              **unused) -> Any:
        """Stack received tensor to elastic aggregator buffer.
        """
        state = self.state[p]
        if 'received_params' not in state:
            state['received_params'] = []

        # stack the instances and received_params
        r_p["instances"] = r_info.instances  # type: ignore
        state['received_params'].append(r_p)

    def _merge_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]
        for key in group['pipe_keys']:
            if key in state:
                if key == "param":
                    # `param` is the general name of the index key tensor.
                    # In most cases, it is the parameters of network or the buffer stored
                    # in the network. Sometimes, you can specify any tensor as `param`.
                    r_p = state[key]
                    if p.requires_grad:
                        # If `param` required grad, we take it as a learnable parameter
                        # of network. In this case, we will calculate the gradient and
                        # copy it to the grad attribute.

                        # NOTE: The received parameters are the updated one.
                        # We should use the original p to sub the received one, yielding the
                        # correct gradient.
                        grad = self._elastic_update(
                                p-r_p, state['importance'], group["quantile"])
                        if p.grad is not None:
                            p.grad.copy_(grad)
                        else:
                            p.grad = grad
                    else:
                        # If `param` does not require gradient, we regard it as the buffer or
                        # user defined tensor, such as the mean and var in batch norm layer.
                        # In this case, we will directly copy the aggregated tensor to cover
                        # the original one.
                        p.copy_(r_p)
                else:
                    # Some inner state is unchanged. Such as `momentum_buffer` if you have specified.
                    # They will unpack to target object from state.
                    ...

    def _stack_aggregate(self, p: Tensor, group: Dict):
        """Aggregate the stack buffer.
        """
        state = self.state[p]

        def aggregate(dl, k, t) -> Tensor:
            return torch.stack(
                [data[k] * (data['instances'] / t)
                 for data in dl],
                dim=0).sum(dim=0, keepdim=False)

        total_instances = sum(
            [data['instances'] for data in state['received_params']])

        for key in group['pipe_keys']:
            if key in state["received_params"][0]:
                r_p = aggregate(
                    state["received_params"], key, total_instances)
                if key == "param":
                    # `param` is the general name of the index key tensor.
                    # In most cases, it is the parameters of network or the buffer stored
                    # in the network. Sometimes, you can specify any tensor as `param`.
                    if p.requires_grad:
                        # If `param` required grad, we take it as a learnable parameter
                        # of network. In this case, we will calculate the gradient and
                        # copy it to the grad attribute.

                        # NOTE: The received parameters are the updated one.
                        # We should use the original p to sub the received one, yielding the
                        # correct gradient.

                        new_imp = aggregate(
                            state["received_params"], "importance", total_instances)
                        grad = self._elastic_update(
                            p-r_p, new_imp, group["quantile"])
                        if p.grad is None:
                            p.grad = grad
                        else:
                            p.grad.copy_(grad)
                    else:
                        # If `param` does not require gradient, we regard it as the buffer or
                        # user defined tensor, such as the mean and var in batch norm layer.
                        # In this case, we will directly copy the aggregated tensor to cover
                        # the original one.
                        p.copy_(r_p)
                else:
                    # Some inner state is unchanged. Such as `momentum_buffer` if you have specified.
                    # They will unpack to target object from state.
                    # Different with merged one, which already calculate the inner state in the
                    # step() process. Stack one must calculate them manually here.
                    state[key] = r_p

    def _elastic_update(self,
                        grad: Tensor,
                        importance: Tensor,
                        quantile: float):
        """Elastic update the gradients with respect to the importance.
        """
        norm_importance = importance / (importance.max() + 1e-13)
        weight = 1 + quantile - norm_importance

        return grad * weight

