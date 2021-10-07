# @Author            : FederalLab
# @Date              : 2021-09-25 16:52:50
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:52:50
# Copyright (c) FederalLab. All rights reserved.
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

from .paillier import Ciphertext, PrivateKey, long_to_float, paillier_dec


def load_param_states(optim: Optimizer, param_states: Dict[Tensor, Any]):
    r"""Loads the state of parameters in optim.

    Args:
        optim: The optimizer.
        param_states: The dictionary contains state of parameters.
    """
    success = 0
    fail = 0
    for group in optim.param_groups:
        for p in group['params']:
            if p in param_states:
                param_state = param_states[p]
                if p.requires_grad:
                    p.grad = param_state['grad']
                optim.state[p].update(param_state)
                success += 1
            else:
                fail += 1
    if fail > 0:
        warnings.warn(f'Load param state for {success} params, '
                      f'{fail} params are ignored.')


def average_aggregation(data_list: List[Dict[Tensor, Any]],
                        meta_list: Optional[Any] = None,
                        optim_list: Optional[Any] = None):
    param_states = defaultdict(dict)

    def aggregate(data: List[Tensor]):
        return torch.stack(data, dim=0).float().mean(dim=0, keepdim=False)

    # get all params
    params = set()
    for data in data_list:
        for p in data:
            params.add(p)
    params = list(params)

    for p in params:
        param_state = param_states[p]
        for data in data_list:
            if p in data:
                state = data[p]
                for k in state:
                    if state[k] is None:
                        continue
                    if k not in param_state:
                        param_state[k] = []
                    param_state[k].append(state[k])

    for p, state in param_states.items():
        for k, v in state.items():
            state[k] = aggregate(v)
        if p.requires_grad:
            state['grad'] = p - state['param']
        else:
            p.copy_(state['param'])
    if optim_list:
        if not isinstance(optim_list, list):
            optim_list = [
                optim_list,
            ]
        for optim in optim_list:
            load_param_states(optim, param_states)
    return param_states


def naive_aggregation(data_list: List[Dict[Tensor, Any]],
                      meta_list: Any,
                      optim_list: Optional[Any] = None):

    assert len(data_list) == len(meta_list)

    param_states = defaultdict(dict)

    def aggregate(data: List[Tensor], weight: List[float]):
        w_data = None
        for d, w in zip(data, weight):
            if w_data is None:
                w_data = d * w
            else:
                w_data += d * w
        return w_data

    # get all params
    params = set()
    for data in data_list:
        for p in data:
            params.add(p)
    params = list(params)

    for p in params:
        param_state = param_states[p]
        total_instances = 0
        for data, meta in zip(data_list, meta_list):
            if p in data:
                total_instances += meta['instances']

        assert total_instances > 0

        for data, meta in zip(data_list, meta_list):
            if p in data:
                weight = meta['instances'] / total_instances
                state = data[p]
                for k in state:
                    if state[k] is None:
                        continue
                    if k not in param_state:
                        param_state[k] = []
                    param_state[k].append(state[k])
                if 'weight' not in param_state:
                    param_state['weight'] = []
                param_state['weight'].append(weight)

    for p, state in param_states.items():
        for k, v in state.items():
            if k != 'weight':
                state[k] = aggregate(v, state['weight'])

        # remove useless weight
        del state['weight']

        if p.requires_grad:
            state['grad'] = p - state['param']
        else:
            p.copy_(state['param'])
    if optim_list:
        if not isinstance(optim_list, list):
            optim_list = [
                optim_list,
            ]
        for optim in optim_list:
            load_param_states(optim, param_states)
    return param_states


def elastic_aggregation(data_list: List[Dict[Tensor, Any]],
                        meta_list: Any,
                        quantile: float = 0.5,
                        optim_list: Optional[Any] = None):

    assert len(data_list) == len(meta_list)

    param_states = defaultdict(dict)

    def aggregate(data: List[Tensor], weight: List[float]):
        w_data = None
        for d, w in zip(data, weight):
            if w_data is None:
                w_data = d * w
            else:
                w_data += d * w
        return w_data

    # get all params
    params = set()
    for data in data_list:
        for p in data:
            params.add(p)
    params = list(params)

    for p in params:
        param_state = param_states[p]
        total_instances = 0
        for data, meta in zip(data_list, meta_list):
            if p in data:
                total_instances += meta['instances']

        assert total_instances > 0

        for data, meta in zip(data_list, meta_list):
            if p in data:
                weight = meta['instances'] / total_instances
                state = data[p]
                for k in state:
                    if state[k] is None:
                        continue
                    if k not in param_state:
                        param_state[k] = []
                    param_state[k].append(state[k])
                if 'weight' not in param_state:
                    param_state['weight'] = []
                param_state['weight'].append(weight)

    for p, state in param_states.items():
        for k, v in state.items():
            if k != 'weight':
                state[k] = aggregate(v, state['weight'])

        # remove useless weight
        del state['weight']

        if p.requires_grad:
            assert 'importance' in state, \
                'elastic aggregation requires `importance` state.'
            norm_importance = state['importance'] / state['importance'].max()
            weight = 1 + quantile - norm_importance
            grad = p - state['param']
            state['grad'] = grad * weight
        else:
            p.copy_(state['param'])
        if 'importance' in state:
            del state['importance']
    if optim_list:
        if not isinstance(optim_list, list):
            optim_list = [
                optim_list,
            ]
        for optim in optim_list:
            load_param_states(optim, param_states)
    return param_states


def paillier_aggregation(data_list: List[Dict[Tensor, Any]],
                         private_key: Union[str, PrivateKey],
                         meta_list: Optional[Any] = None,
                         optim_list: Optional[Any] = None):
    if isinstance(private_key, str):
        private_key = PrivateKey.load(private_key)

    def decode(c1, c2, received_numbers):
        c = Ciphertext(c1, c2)
        v = paillier_dec(private_key, c)  # type: ignore
        return long_to_float(private_key, v, received_numbers)  # type: ignore

    param_states = defaultdict(dict)

    def aggregate(data: List[Tensor]):
        return torch.stack(data, dim=0).float().mean(dim=0, keepdim=False)

    # get all params
    params = set()
    for data in data_list:
        for p in data:
            params.add(p)
    params = list(params)

    for p in params:
        param_state = param_states[p]
        received_numbers = 0
        for data in data_list:
            if p in data:
                received_numbers += 1
                state = data[p]
                for k in state:
                    if state[k] is None:
                        continue
                    if k not in param_state:
                        param_state[k] = []

                    param_state[k].append(state[k])

        param_state['received_numbers'] = received_numbers

    for p, state in param_states.items():
        received_numbers = state.pop('received_numbers')
        for k, v in state.items():
            state[k] = aggregate(v)
        # decode
        keys = [k[:-3] for k in state if k.endswith('_c1')]
        for k in keys:
            c1 = state[f'{k}_c1']
            c2 = state[f'{k}_c2']
            del state[f'{k}_c1']
            del state[f'{k}_c2']
            data = decode(c1, c2, received_numbers)
            state[k] = data[:p.numel()].reshape_as(p)

        if p.requires_grad:
            state['grad'] = p - state['param']
        else:
            p.copy_(state['param'])
    if optim_list:
        if not isinstance(optim_list, list):
            optim_list = [
                optim_list,
            ]
        for optim in optim_list:
            load_param_states(optim, param_states)
    return param_states
