from .aggregator import Aggregator
import torch
from typing import Any, List, Dict
from openfed import PACKAGES

class NaiveAggregator(Aggregator):
    """朴素聚合是一种针对平均聚合的简单的改进。这种方式被广泛运用于各种任务中。
    """

    def __init__(self, params, enable_merge: bool = True):
        """需要客户端返回train_instances键值。
        """
        necessary_keys: List[str] = ['train_instances']
        reset_keys: List[str] = ["step", "received_params"]
        defaults = dict()
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

        r_p = received_params[p]['param']

        if p.requires_grad:
            # update in grad
            # 注意：这里grad存的是更新后的模型，不是梯度！
            # step()函数会统一将其转换成梯度。
            if not p.grad:
                p.grad = torch.zeros_like(p)
            r_p = (p.grad * step + r_p * train_instances) / \
                (step + train_instances)
            p.grad.copy_(r_p)
        else:
            # modify p itself
            new_p = (p * step + r_p * train_instances) / \
                (step + train_instances)
            p.copy_(new_p)
        state['step'] += train_instances

    def append(self, p: torch.Tensor, received_params: PACKAGES, received_info: Dict, group) -> Any:
        train_instances = received_info['train_instances']
        state = self.state[p]
        r_p = received_params[p]['param']
        if 'received_params' not in state:
            state['received_params'] = []
        state['received_params'].append([r_p, train_instances])

    def _merge_step(self, p: torch.Tensor, state: Dict, group):
        # 对于这种方式下，如果p是梯度可更新的参数，则将梯度计算出来
        # 否则保持不变即可。
        if p.requires_grad:
            # 记住！是p-grad，不是grad-p
            # grad保存的是训练以后的参数！而不是梯度！
            p.grad.copy_(p-p.grad)

    def _append_step(self, p: torch.Tensor, state: Dict, group):
        total = 0
        # 增加下述代码的可读性
        for _, ti in state["received_params"]:
            total += ti
        for i, (n_p, t_i) in enumerate(state['received_params']):
            w = t_i / total
            n_p = n_p * w
            if i == 0:
                new_p = n_p
            else:
                new_p += n_p

        if p.requires_grad:
            if not p.grad:
                p.grad = p-new_p
            else:
                p.grad.copy_(p-new_p)
        else:
            p.copy_(new_p)
