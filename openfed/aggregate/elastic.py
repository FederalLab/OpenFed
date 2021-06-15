from .aggregator import Aggregator
import torch
from typing import Any, List, Dict
from torch import Tensor
from openfed.utils.types import PACKAGES


class ElasticAggregator(Aggregator):
    """弹性聚合是一种能够根据客户端数据的特性进行参数控制的方法。
    """

    def __init__(self, params, quantile=0.5, lagecy: bool = True):
        """需要客户端返回train_instances键值。
        """
        info_keys: List[str] = ['train_instances']
        # 会缓存step、received params
        aux_keys: List[str] = [
            "step", "received_params", "param", "importance"]

        defaults = dict(quantile=quantile,)
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

        if 'param' not in state:
            state['param'] = torch.zeros_like(p)
        param = state['param']

        if p.requires_grad:
            # 在这种情况下，每一个参数对应两个值
            r_p = received_params[p]['param']
            r_i = received_params[p]['importance']

            state['param'] = (param * step + r_p * train_instances) / \
                (step + train_instances)

            if 'importance' not in state:
                state['importance'] = torch.zeros_like(p)
            importance = state['importance'
                               ]
            state['importance'] = (importance * step + r_i *
                                   train_instances) / (step + train_instances)
        else:
            # 这种情况下，每一个参数对应一个值。
            # 只有需要梯度更新的参数，才会有重要性参数。
            r_p = received_params[p]['param']
            # modify p itself
            state['param'] = (param * step + r_p * train_instances) / \
                (step + train_instances)
        state['step'] += train_instances

    def stack(self, p: Tensor, received_params: PACKAGES, received_info: Dict, **unused) -> Any:
        train_instances = received_info['train_instances']
        state = self.state[p]
        if 'received_params' not in state:
            state['received_params'] = []
        # 如果需要梯度的话，每一个参数对应两个返回值
        # 一个是更新后的参数，一个是对应的重要性因子
        # 否则的话，仍然是一个值

        if p.requires_grad:
            r_p, r_i = received_params[p]['param'], received_params[p]['importance']
            state['received_params'].append(
                [r_p, r_i, train_instances])
        else:
            r_p = received_params[p]['param']
            state['received_params'].append(
                [r_p, train_instances])

    def _merge_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]
        # 对于这种方式下，如果p是梯度可更新的参数，则将梯度计算出来
        # 否则保持不变即可。
        new_param = state['param']
        if p.requires_grad:
            # 记住！是p-grad，不是grad-p
            # grad保存的是训练以后的参数！而不是梯度！
            p.grad.copy_(self._elastic_update(
                p-new_param, state['importance'], group['quantile']))
        else:
            p.copy_(new_param)

    def _stack_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]
        total_instances = 0
        for _, _, instances in state["received_params"]:
            total_instances += instances

        new_param = 0
        new_imp = 0
        for data in state['received_params']:
            if p.requires_grad:
                param, importance, instances = data
                weight = instances / total_instances
                param = param * weight
                importance = importance * weight
                new_imp += importance
                new_param += param
            else:
                param, instances = data
                weight = instances / total_instances
                param = param * weight
                new_param += param

        if p.requires_grad:
            grad = self._elastic_update(
                p-new_param, new_imp, group["quantile"])
            if not p.grad:
                p.grad = grad
            else:
                p.grad.copy_(grad)
        else:
            p.copy_(new_param)

    def _elastic_update(self, grad: Tensor, importance: Tensor, quantile: float):
        norm_importance = importance / (importance.max() + 1e-13)
        weight = 1 + quantile - norm_importance

        return grad * weight
