from typing import Any, Dict, List, Union

from torch import Tensor

from .aggregator import Aggregator


class ElasticAggregator(Aggregator):
    """弹性聚合是一种能够根据客户端数据的特性进行参数控制的方法。
    """

    def __init__(self, params, other_keys: Union[str, List[str]], quantile=0.5, lagecy: bool = True):
        """需要客户端返回train_instances键值。
        """
        if not (0 < quantile < 1.0):
            raise ValueError("quantile must be between 0 and 1")

        if isinstance(other_keys, str):
            other_keys = [other_keys]

        info_keys: List[str] = ['train_instances']
        # 会缓存step、received params
        aux_keys: List[str] = [
            "step", "received_params", "param", "importance"]

        for k in other_keys:
            if k in aux_keys:
                raise ValueError(f"Duplicate key: {k}")

        aux_keys.extend(other_keys)

        defaults = dict(quantile=quantile,)
        super().__init__(
            params,
            defaults,
            info_keys=info_keys,
            aux_keys=aux_keys,
            lagecy=lagecy)

    def merge(self, p: Tensor, r_p: Dict[str, Tensor], received_info: Dict, group: Dict) -> Any:
        train_instances = received_info['train_instances']
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        for key in group['aux_keys']:
            if key in r_p:
                if key not in state:
                    state[key] = r_p[key]
                else:
                    state[key] = (state[key] * step + r_p[key] * train_instances) / \
                        (step + train_instances)
        state['step'] += train_instances

    def stack(self, p: Tensor, r_p: Dict[str, Tensor], received_info: Dict, **unused) -> Any:
        state = self.state[p]
        if 'received_params' not in state:
            state['received_params'] = []

        r_p["train_instances"] = received_info['train_instances']
        state['received_params'].append(r_p)

    def _merge_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]
        # 对于这种方式下，如果p是梯度可更新的参数，则将梯度计算出来
        # 否则保持不变即可。

        for key in group['aux_keys']:
            if key in state:
                new_p = state[key]
                if key == "param":
                    if p.requires_grad:
                        # 记住！是p-grad，不是grad-p
                        # grad保存的是训练以后的参数！而不是梯度！
                        p.grad.copy_(self._elastic_update(
                            p-new_p, state['importance'], group['quantile']))
                    else:
                        p.copy_(new_p)
                else:
                    # 保持不变，啥也不需要做
                    ...

    def _stack_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]

        def aggregate(dl, k, t):
            l = 0
            for data in dl:
                a, b = data[k], data['train_instances']
                w = b / t
                p = a * w
                l += p
            return l

        total_instances = 0
        for data in state["received_params"]:
            instances = data["train_instances"]
            total_instances += instances

        for key in group['aux_keys']:
            if key in state["received_params"][0]:
                new_p = aggregate(
                    state["received_params"], key, total_instances)
                if key == "param":
                    if p.requires_grad:
                        new_imp = aggregate(
                            state["received_params"], "importance", total_instances)
                        grad = self._elastic_update(
                            p-new_p, new_imp, group["quantile"])
                        if not p.grad:
                            p.grad = grad
                        else:
                            p.grad.copy_(grad)
                    else:
                        p.copy_(new_p)
                else:
                    # 把值写入相关的状态
                    state[key] = new_p

    def _elastic_update(self, grad: Tensor, importance: Tensor, quantile: float):
        norm_importance = importance / (importance.max() + 1e-13)
        weight = 1 + quantile - norm_importance

        return grad * weight
