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


import warnings
from collections import abc as container_abcs
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List

import torch
from openfed.common import Package, TaskInfo
from openfed.utils import convert_to_list
from torch import Tensor


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Agg."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Agg(Package):
    r"""Base class for Aggregation.

    Aggregator collects all received tensor and related task information.
    At each round end, it will aggregate gradient from received models and
    save them to `param.grad` attribute.

    .. note::
        Aggregator only calculate gradients for params, just like `loss.backward`.
    It is never to modify any parameters directly. The update operation is still
    left for `optim` or `fed_optim`.

    .. note::
        Aggregator needs be coupled with ``Reducer``, which will reduce the received
    task information and yields the task-related training or testing results.
    """
    # Used by Reducer, it will be cleared automatically once `Reducer.reduce()` callled.
    task_info_buffer: List[TaskInfo]
    # Whether the state has been cached.
    empty_state_cache: bool

    def __init__(self,
                 params,
                 defaults: Dict[Any, Any]):
        """
        Args:
            info_keys: The keys must be returned in the `received_info`. 
                It is used to calculate the weight of each clients or any other statistic
                information about each client.
            pipe_keys: The keys need to be returned in the `received_params`. 
                It is used to calculate the inner state of fed_optim, such as `scaffold`.
            legacy: If `True`, stack received data into buffer directly. 
                If `False`, merge received data into buffer. The latter one only take
                const memory cost, but the formmer one will cost a O(n) memory cost.
                However, the merge way will discard the accurate value of each received model,
                which may not be desired at some times.
        """
        # add info_keys to defaults
        defaults['info_keys'] = defaults.get('info_keys', [])
        defaults['pipe_keys'] = defaults.get('pipe_keys', [])
        defaults['legacy'] = defaults.get('legacy', False)

        self.defaults = defaults

        if isinstance(params, Tensor):
            raise TypeError("params argument given to the agg should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("agg got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        self.task_info_buffer = []
        self.empty_state_cache = True

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def state_dict(self):
        r"""Returns the state of the agg as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current aggregation state. Its content
            differs between agg classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the agg state.

        Args:
            state_dict (dict): agg state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of agg's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value) # type: ignore
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)  # type: ignore
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`Tensor` s to zero.
        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` aggregators have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Agg` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Agg` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific aggregation options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('agg parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, Tensor):
                raise TypeError("agg can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required aggregation parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("agg contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def aggregate(self, clear_buffer: bool = True):
        r"""Performs a single aggregation step on all received data.
        It should be called at the end of round.

        Args: 
            clear_buffer: If `True`, it will clear the cached data.

        .. note::
            `aggregate()` has a similar function as `backward()`, which only calculate
            the gradient and store them in `p.grad`, but do not attempt to modify the 
            parameters itself.
        """
        if not self.empty_state_cache:
            for group in self.param_groups:
                legacy = group['legacy']
                for p in group["params"]:
                    [self._stack_aggregate(
                        p, group) if legacy else self._merge_aggregate(p, group)]
        if clear_buffer:
            self.clear_buffer()

    def step(self, received_params: Dict[Tensor, Dict[str, Tensor]], received_info: TaskInfo) -> None:
        """Called once received a new data.
        Args:
            received_params: A tensor indexed dictionary, which usually returned by `delivery.indexed_tensor_packages`. Each items must contain all `pipe_keys`.
            received_info: The received task information, which must contain all `info_keys`.

        .. note::
            `step()` only caches the received info, but does anything else. You should call 
            `aggregate()` to compute the final gradient for each parameter.
        """
        if received_params:
            for group in self.param_groups:
                # info keys is the necessary keys for computing the aggregated tensor.
                # If not given, raise an error.
                for key in group['info_keys']:
                    assert key in received_info, f"{key} is required, but not given."

                for p in group["params"]:
                    if p in received_params:
                        if group['legacy']:
                            self.stack(
                                p,
                                received_params[p],
                                r_info=received_info,
                                group=group)
                        else:
                            self.merge(
                                p,
                                received_params[p],
                                r_info=received_info,
                                group=group)
            self.empty_state_cache = False
        else:
            self.empty_state_cache = True

        # Cache received info to task info.
        # The task_info_buffer will be accessed by Reducer, and get the final results.
        self.task_info_buffer.append(received_info)

    def merge(self, p: Tensor, r_p: Dict[str, Tensor], r_info: TaskInfo, group: Dict) -> Any:
        """Merge a received data to buffer.
        Args:
            p: The tensor used as index.
            r_p: The received data dictionary.
            r_info: The received task info dictionary.
            group: The group `p` belongs to. It contains other necessary hyper parameters.
        """
        raise NotImplementedError

    def _merge_aggregate(self, p: Tensor, group: Dict) -> None:
        """Aggregate operation on merged buffer.
        Args:
            p: The parameter attached gradient to.
            group: The group contains necessary hyper-parameters.
        """
        raise NotImplementedError

    def stack(self, p: Tensor, r_p: Dict[str, Tensor], r_info: TaskInfo, group: Dict) -> Any:
        """Stack a received data to buffer.
        Args:
            p: The tensor used as index.
            r_p: The received data dictionary.
            r_info: The received task info dictionary.
            group: The group `p` belongs to. It contains other necessary hyper parameters.
        """
        raise NotImplementedError

    def _stack_aggregate(self, p: Tensor, group: Dict) -> None:
        """Aggregate operation on stack buffer.
        Args:
            p: The parameter attached gradient to.
            group: The group contains necessary hyper-parameters.
        """
        raise NotImplementedError

    def clear_buffer(self):
        super().clear_buffer()
        self.empty_state_cache = True

class AverageAgg(Agg):
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


class ElasticAgg(Agg):
    """Elastic Aggregation: Aggregate received tensor in a data-aware way.

    .. warn::
        ElasticAgg must be coupled with `ElasticPenalizer`.
    """

    def __init__(self,
                 params,
                 quantile: float = 0.5,
                 legacy: bool = False,
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
            legacy=legacy
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


class NaiveAgg(Agg):
    """Naive Aggregator: Aggregate received tensor in a naive weighted sum.

    Naive Aggregator has been widely used in FedAvg and other algorithms.
    """

    def __init__(self,
                 params,
                 other_keys: List[str] = None,
                 legacy: bool = False):
        other_keys = convert_to_list(other_keys) or []
        info_keys = ['instances']
        pipe_keys = list(
            set(["step", "received_params", "param"] + other_keys))

        defaults = dict(
            info_keys=info_keys,
            pipe_keys=pipe_keys,
            legacy=legacy,
        )
        super().__init__(params, defaults)

    def merge(self,
              p     : Tensor,
              r_p   : Dict[str, Tensor],
              r_info: TaskInfo,
              group : Dict) -> Any:
        """Merge received tensor to naive aggregator buffer.
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
                        grad = p - r_p
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
            return torch.stack([data[k] * (data['instances'] / t) for data in dl], dim=0).sum(dim=0, keepdim=False)

        total_instances = sum(
            [data['instances'] for data in state['received_params']])

        for key in group['pipe_keys']:
            if key in state['received_params'][0]:
                r_p = aggregate(
                    state['received_params'], key, total_instances)
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
                    state[key] = r_p


aggregators = [
    Agg, ElasticAgg, AverageAgg, NaiveAgg
]
