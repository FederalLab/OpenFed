from typing import Any, List

import torch
from openfed.utils.types import PACKAGES
from torch import Tensor

from .aggregator import Aggregator


class AverageAggregator(Aggregator):
    """平均聚合是一种最简单的聚合方式，它直接将返回的参数取平均。
    """

    def __init__(self, params, lagecy: bool = True):
        # 不需要额外的信息
        info_keys: List[str] = []
        # 会缓存step、received params
        aux_keys: List[str] = ["step", "received_params", "param"]
        defaults = dict()
        super().__init__(
            params,
            defaults,
            info_keys=info_keys,
            aux_keys=aux_keys,
            lagecy=lagecy)

    def merge(self, p: Tensor, received_params: PACKAGES, **unused) -> Any:
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        r_p = received_params[p]['param']

        if 'param' not in state:
            state['param'] = r_p
        else:
            state['param'] = (state['param'] * step + r_p) / (step + 1)
        state['step'] += 1

    def stack(self, p: torch.Tensor, received_params: PACKAGES, **unused) -> Any:
        state = self.state[p]

        if 'received_params' not in state:
            state['received_params'] = []
        state['received_params'].append(received_params['param'])

    def _merge_aggregate(self, p: torch.Tensor, **unused):
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

    def _stack_aggregate(self, p: torch.Tensor, **unused):
        # check here
        state = self.state[p]
        new_p = torch.cat(state["received_params"], dim=0).mean(dim=0)
        if p.requires_grad:
            if not p.grad:
                p.grad = p-new_p
            else:
                p.grad.copy_(p-new_p)
        else:
            p.copy_(new_p)
