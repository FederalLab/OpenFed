from typing import Any, Dict, List

import torch
from openfed.utils.types import PACKAGES
from torch import Tensor
from .aggregator import Aggregator


class NaiveAggregator(Aggregator):
    """朴素聚合是一种针对平均聚合的简单的改进。这种方式被广泛运用于各种任务中。
    """

    def __init__(self, params, lagecy: bool = True):
        """需要客户端返回train_instances键值。
        """
        info_keys: List[str] = ['train_instances']
        # 会缓存step、received params
        aux_keys: List[str] = ["step", "received_params", "param"]
        defaults = dict()
        super().__init__(
            params,
            defaults,
            info_keys=info_keys,
            aux_keys=aux_keys,
            lagecy=lagecy)

    def merge(self, p: Tensor, received_params: PACKAGES, received_info: Dict, **unused) -> Any:
        train_instances = received_info['train_instances']
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        r_p = received_params[p]['param']

        if 'param' not in state:
            state['param'] = r_p
        else:
            state['param'] = (state['param'] * step +
                              r_p * train_instances) / (step + train_instances)
        state['step'] += train_instances

    def stack(self, p: Tensor, received_params: PACKAGES, received_info: Dict, **unused) -> Any:
        train_instances = received_info['train_instances']
        state = self.state[p]
        r_p = received_params[p]['param']
        if 'received_params' not in state:
            state['received_params'] = []
        state['received_params'].append([r_p, train_instances])

    def _merge_aggregate(self, p: Tensor, **unused):
        state = self.state[p]
        # 对于这种方式下，如果p是梯度可更新的参数，则将梯度计算出来
        # 否则保持不变即可。
        new_p = state['param']
        if p.requires_grad:
            # 记住！是p - new_p，不是new_p - p
            # grad保存的是训练以后的参数！而不是梯度！
            p.grad.copy_(p - new_p)
        else:
            p.copy_(new_p)

    def _append_step(self, p: Tensor, **unused):
        state = self.state[p]

        total_instances = 0
        for _, instances in state["received_params"]:
            total_instances += instances

        new_param = 0
        for param, instances in state['received_params']:
            w = instances / total_instances
            param = param * w
            new_param += param

        if p.requires_grad:
            if not p.grad:
                p.grad = p-new_param
            else:
                p.grad.copy_(p-new_param)
        else:
            p.copy_(new_param)
