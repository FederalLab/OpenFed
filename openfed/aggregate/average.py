from typing import Any, Dict, List, Union

import torch
from torch import Tensor

from .aggregator import Aggregator


class AverageAggregator(Aggregator):
    """平均聚合是一种最简单的聚合方式，它直接将返回的参数取平均。
    """

    def __init__(self, params, other_keys: Union[str, List[str]], lagecy: bool = True):
        """other_keys 是那些额外的键值，比如用于同步optimizer的状态，这时候就可以把相关的dict传入。
        """
        if isinstance(other_keys, str):
            other_keys = [other_keys]

        # 不需要额外的信息
        info_keys: List[str] = []
        # 会缓存step、received params
        aux_keys: List[str] = ["step", "received_params", "param"]

        for k in other_keys:
            if k in aux_keys:
                raise ValueError(f"Duplicate key: {k}")

        aux_keys.extend(other_keys)
        defaults = dict()
        super().__init__(
            params,
            defaults,
            info_keys=info_keys,
            aux_keys=aux_keys,
            lagecy=lagecy)

    def merge(self, p: Tensor, r_p: Dict[str, Tensor], received_info: Dict, group: Dict) -> Any:
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        for key in group['aux_keys']:
            if key in r_p:
                # 有些键值可能是缺失的在一些情况下。
                # 如果出现相关的键值，就统计起来
                if key not in state:
                    state[key] = r_p[key]
                else:
                    state[key] = (state[key] * step + r_p[key]) / (step + 1)
        state['step'] += 1

    def stack(self, p: torch.Tensor, r_p: Dict[str, Tensor], **unused) -> Any:
        state = self.state[p]

        if 'received_params' not in state:
            state['received_params'] = []
        state['received_params'].append(r_p)

    def _merge_aggregate(self, p: torch.Tensor, group: Dict):
        state = self.state[p]
        # 对于这种方式下，如果p是梯度可更新的参数，则将梯度计算出来
        # 否则保持不变即可
        aux_keys = group['aux_keys']

        for key in aux_keys:
            if key in state:
                new_p = state[key]
                if key == "param":
                    if p.requires_grad:
                        # 记住！是p - new_p，不是new_p - p
                        # grad保存的是训练以后的参数！而不是梯度！
                        p.grad.copy_(p - new_p)
                    else:
                        p.copy_(new_p)
                else:
                    # 保持不变，这些都是辅助变量，让他们保存在字典里。
                    # 让unpack_state函数正常调用即可。
                    ...

    def _stack_aggregate(self, p: torch.Tensor, group: Dict):
        state = self.state[p]

        def aggregate(dl, k):
            l = []
            for data in dl:
                l.append(data[k])
            return torch.stack(l, dim=0).mean(dim=0, keepdims=False)

        aux_keys = group['aux_keys']
        for key in aux_keys:
            if key in state['received_params'][0]:
                new_p = aggregate(state["received_params"], key)
                if key == "param":
                    if p.requires_grad:
                        if not p.grad:
                            p.grad = p-new_p
                        else:
                            p.grad.copy_(p-new_p)
                    else:
                        p.copy_(new_p)
                else:
                    # 把值写入相关状态
                    state[key] = new_p
