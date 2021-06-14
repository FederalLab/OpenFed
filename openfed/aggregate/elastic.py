from .aggregator import Aggregator
import torch
from typing import Any, List, Dict, Union
from torch import Tensor
from openfed import PACKAGES


class ElasticAggregator(Aggregator):
    """弹性聚合是一种能够根据客户端数据的特性进行参数控制的方法。
    """

    def __init__(self, params, quantile=0.5, enable_merge: bool = True):
        """需要客户端返回train_instances键值。
        """
        necessary_keys: List[str] = ['train_instances']
        reset_keys: List[str] = ["step", "importance", "received_params"]
        defaults = dict(quantile=quantile,)
        super().__init__(
            params,
            defaults,
            necessary_keys=necessary_keys,
            reset_keys=reset_keys,
            enable_merge=enable_merge)

    def merge(self, p: torch.Tensor, received_params: PACKAGES, received_info: Dict, group) -> Any:
        train_instances = received_info['train_instances']
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        if p.requires_grad:
            # 在这种情况下，每一个参数对应两个值
            r_p = received_params[p]['param']
            r_i = received_params[p]['importance']

            # update in grad
            # 注意：这里grad存的是更新后的模型，不是梯度！
            # step()函数会统一将其转换成梯度。
            if not p.grad:
                p.grad = torch.zeros_like(p)
            r_p = (p.grad * step + r_p * train_instances) / \
                (step + train_instances)
            p.grad.copy_(r_p)

            if 'importance' not in state:
                state['importance'] = torch.zeros_like(p)
            r_i = (state['importance'] * step + r_i *
                   train_instances) / (step + train_instances)
            state["importance"].copy_(r_i)
        else:
            # 这种情况下，每一个参数对应一个值。
            # 只有需要梯度更新的参数，才会有重要性参数。
            r_p = received_params[p]['param']
            # modify p itself
            new_p = (p * step + r_p * train_instances) / \
                (step + train_instances)
            p.copy_(new_p)
        state['step'] += train_instances

    def append(self, p: torch.Tensor, received_params: PACKAGES, received_info: Dict, group) -> Any:
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

    def _merge_step(self, p: torch.Tensor, state: Dict, group):
        # 对于这种方式下，如果p是梯度可更新的参数，则将梯度计算出来
        # 否则保持不变即可。
        if p.requires_grad:
            # 记住！是p-grad，不是grad-p
            # grad保存的是训练以后的参数！而不是梯度！
            p.grad.copy_(self._elastic_update(
                p-p.grad, state['importance'], group['quantile']))

    def _append_step(self, p: torch.Tensor, state: Dict, group):
        # TODO:加强以下代码的可读性
        total = 0
        for _, _, ti in state["received_params"]:
            total += ti
        for i, data in enumerate(state['received_params']):
            if p.requires_grad:
                n_p, i_w, t_i = data
                w = t_i / total
                n_p = n_p * w
                i_w = i_w * w
                if i == 0:
                    new_p = n_p
                    new_i = i_w
                else:
                    new_p += n_p
                    new_i += i_w
            else:
                n_p, t_i = data
                w = t_i / total
                n_p = n_p * w
                if i == 0:
                    new_p = n_p
                else:
                    new_p += n_p

        if p.requires_grad:
            grad = self._elastic_update(p-new_p, new_i, group["quantile"])
            if not p.grad:
                p.grad = grad
            else:
                p.grad.copy_(grad)
        else:
            p.copy_(new_p)

    def _elastic_update(self, grad: torch.Tensor, importance: torch.Tensor, quantile: float):
        norm_importance = importance / (importance.max() + 1e-13)
        weight = 1 + quantile - norm_importance

        return grad * weight
