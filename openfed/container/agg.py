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
from typing import Any, Dict, List, Union

import torch
from openfed.common import Package, TaskInfo, Wrapper, Buffer
from openfed.utils import convert_to_list
from torch import Tensor


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Agg."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Agg(Package, Wrapper, Buffer):
    r"""Base class for Agg.
    """
    task_info_buffer: List[TaskInfo]  # Used by Reducer, it will be cleared once .reduce() is callled.

    def __init__(self,
                 params,
                 defaults: Dict,
                 info_keys: List[str],
                 pipe_keys: List[str],
                 keep_keys: List[str] = None,
                 legacy: bool = False):
        """
        Args:
            info_keys: necessary keys saved in returned info dict.
            pipe_keys: other tensor that needed to saved.
            keep_keys: the state which will not be cleared while .clear_buffer() is called
            legacy: if True, just stack received data, otherwise will merge them.
        """
        self.legacy = legacy

        # add info_keys to defaults
        defaults['info_keys'] = convert_to_list(info_keys)
        defaults['pipe_keys'] = convert_to_list(pipe_keys)
        defaults['keep_keys']  = convert_to_list(keep_keys)
        defaults['legacy'] = legacy

        self.defaults = defaults

        if isinstance(params, torch.Tensor):
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
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
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
            if isinstance(value, torch.Tensor):
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
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.
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
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('agg parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
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

    def _check_defaults_keys(self, info_keys: List[str], received_info: TaskInfo):
        for key in info_keys:
            if key not in received_info.info_dict:
                raise KeyError(f"{key} is needed, but not returned.")

    def aggregate(self, clear_buffer: bool = True):
        r"""Performs a single aggregation step (parameter update).

        Args: 
            clear_buffer: if True, will clear the cached data.
        """
        for group in self.param_groups:
            legacy = group['legacy']
            for p in group["params"]:
                if legacy:
                    self._stack_aggregate(p, group=group)
                else:
                    self._merge_aggregate(p, group=group)

        if clear_buffer:
            self.clear_buffer()

    def step(self, received_params: Dict[str, Dict[str, Tensor]], received_info: TaskInfo) -> None:
        """Add a new received data.
        """
        for group in self.param_groups:
            self._check_defaults_keys(group['info_keys'], received_info)
            legacy = group['legacy']
            for p in group["params"]:
                if p in received_params:
                    if legacy:
                        self.stack(p, received_params[p],
                                   received_info=received_info, group=group)
                    else:
                        self.merge(p, received_params[p],
                                   received_info=received_info, group=group)
        self.task_info_buffer.append(received_info)

    def merge(self, p: Tensor, r_p: Dict[str, Tensor], received_info: TaskInfo, group: Dict) -> Any:
        raise NotImplementedError

    def _merge_aggregate(self, p: torch.Tensor, group: Dict) -> None:
        raise NotImplementedError

    def stack(self, p: Tensor, r_p: Dict[str, Tensor], received_info: TaskInfo, group: Dict) -> Any:
        raise NotImplementedError

    def _stack_aggregate(self, p: torch.Tensor, group: Dict) -> None:
        raise NotImplementedError

    def unpack(self, key: Tensor, rdict: Dict[str, Any]) -> Dict[str, Tensor]:
        """used for Package.
        """
        state = self.state[key]
        return {key: state[key] for key in rdict}


class AverageAgg(Agg):
    """average all received data directly.
    """

    def __init__(self,
                 params,
                 other_keys: List[str] = None,
                 keep_keys: List[str] = None,
                 legacy: bool = True):
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
            info_keys=info_keys,
            pipe_keys=pipe_keys,
            keep_keys=keep_keys,
            legacy=legacy)

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
                        if p.grad is not None:
                            # NOTE: grad = p - new_p
                            p.grad.copy_(p - new_p)
                        else:
                            p.grad = p - new_p
                    else:
                        p.copy_(new_p)
                else:
                    ...

    def _stack_aggregate(self, p: torch.Tensor, group: Dict):
        state = self.state[p]

        def aggregate(dl, k):
            return torch.stack([data[k] for data in dl], dim=0).mean(dim=0, keepdim=False)

        pipe_keys = group['pipe_keys']
        for key in pipe_keys:
            if key in state['received_params'][0]:
                new_p = aggregate(state["received_params"], key)
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


class ElasticAgg(Agg):
    """A data-award aggregation method.

    This aggregator must be paired with `elastic pipe`.
    """

    def __init__(self, params,
                 other_keys: Union[str, List[str]] = None,
                 keep_keys: List[str] = None,
                 quantile: float = 0.5,
                 legacy: bool = True):

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
            info_keys=info_keys,
            pipe_keys=pipe_keys,
            keep_keys=keep_keys,
            legacy=legacy)

    def merge(self,
              p: Tensor,
              r_p: Dict[str, Tensor],
              received_info: TaskInfo,
              group: Dict) -> Any:
        instances = received_info.instances  # type: ignore
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
              p: Tensor,
              r_p: Dict[str, Tensor],
              received_info: TaskInfo,
              **unused) -> Any:
        state = self.state[p]
        if 'received_params' not in state:
            state['received_params'] = []

        r_p["instances"] = received_info.instances  # type: ignore
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
                w = b / t
                p = a * w
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
                        grad: Tensor,
                        importance: Tensor,
                        quantile: float):
        norm_importance = importance / (importance.max() + 1e-13)
        weight = 1 + quantile - norm_importance

        return grad * weight


class NaiveAgg(Agg):
    """widely used in FedAvg.
    """

    def __init__(self,
                 params,
                 other_keys: Union[str, List[str]] = None,
                 keep_keys: List[str] = None,
                 legacy: bool = True):
        other_keys = [] if other_keys is None else convert_to_list(other_keys)

        info_keys: List[str] = ['instances']
        pipe_keys: List[str] = ["step", "received_params", "param"]

        for k in other_keys:
            if k in pipe_keys:
                raise ValueError(f"Duplicate key: {k}")

        pipe_keys.extend(other_keys)
        defaults = dict()
        super().__init__(
            params,
            defaults,
            info_keys=info_keys,
            pipe_keys=pipe_keys,
            keep_keys=keep_keys,
            legacy=legacy)

    def merge(self,
              p: Tensor,
              r_p: Dict[str, Tensor],
              received_info: TaskInfo,
              group: Dict) -> Any:
        instances = received_info.instances  # type: ignore
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
              p: Tensor,
              r_p: Dict[str, Tensor],
              received_info: TaskInfo,
              **unused) -> Any:
        state = self.state[p]
        if 'received_params' not in state:
            state['received_params'] = []

        r_p["instances"] = received_info.instances  # type: ignore
        state['received_params'].append(r_p)

    def _merge_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]

        for key in group['pipe_keys']:
            if key in state:
                new_p = state[key]
                if key == "param":
                    if p.requires_grad:
                        if p.grad is not None:
                            p.grad.copy_(p - new_p)
                        else:
                            p.grad = p - new_p
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
                w = b / t
                p = a * w
                l.append(p)
            return torch.stack(l, dim=0).sum(dim=0, keepdim=False)

        total_instances = 0
        for data in state["received_params"]:
            instances = data['instances']
            total_instances += instances

        for key in group['pipe_keys']:
            if key in state['received_params'][0]:
                new_p = aggregate(
                    state['received_params'], key, total_instances)
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


aggregators = [
    Agg, ElasticAgg, AverageAgg, NaiveAgg
]