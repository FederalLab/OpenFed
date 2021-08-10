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
from openfed.utils import convert_to_list
from torch import Tensor

from .agg_op import AggOp


class AverageOp(AggOp):
    """Average Aggregation: aggregate received tensor with an average operation.
    """

    def __init__(self,
                 params,
                 other_keys: List[str] = None,
                 legacy: bool = False):
        """
        Args:
            other_keys: The keys you want to track, like `momentum_buffer`, `exp_avg`, `exp_avg_sq`.
        """
        other_keys = convert_to_list(other_keys) or []
        pipe_keys = list(
            set(["step", "received_params", "param"] + other_keys))

        defaults = dict(
            pipe_keys=pipe_keys,
            legacy=legacy,
        )
        super().__init__(params, defaults)

    def merge(self, p: Tensor, r_p: Dict[str, Tensor], r_info: Dict, group: Dict) -> Any:
        """Merge received tensor to average aggregator buffer.
        """
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        for key in group['pipe_keys']:
            if key in r_p:
                # Merge the received tensor to previous stored one
                # Here, a simple average operation over buffer is applied.
                state[key] = r_p[key] if key not in state else (
                    state[key] * step + r_p[key]) / (step + 1)
        state['step'] += 1

    def stack(self, p: Tensor, r_p: Dict[str, Tensor], **unused) -> Any:
        """Stack received tensor to average aggregator buffer.
        """
        state = self.state[p]
        if 'received_params' not in state:
            state['received_params'] = []
        # Simply stack the received tensors to buffer.
        # If will be more flexible if you want to do anything else.
        state['received_params'].append(r_p)

    def _merge_aggregate(self, p: Tensor, group: Dict):
        """Aggregate the merged buffer.
        """
        state = self.state[p]
        pipe_keys = group['pipe_keys']

        for key in pipe_keys:
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
                        grad = p - r_p
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
                    ...

    def _stack_aggregate(self, p: Tensor, group: Dict):
        """Aggregate the stack buffer.
        """
        def aggregate(dl, k):
            return torch.stack([data[k] for data in dl], dim=0).mean(dim=0, keepdim=False)

        state = self.state[p]
        pipe_keys = group['pipe_keys']
        for key in pipe_keys:
            if key in state['received_params'][0]:
                r_p = aggregate(state["received_params"], key)
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
                        grad = p-r_p
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
