from typing import Any, Dict, List, Union

from torch import Tensor

from ..base import Aggregator


class NaiveAggregator(Aggregator):
    """widely used in FedAvg.
    """

    def __init__(self,
                 params,
                 other_keys: Union[str, List[str]] = None,
                 legacy: bool = True):
        if isinstance(other_keys, str):
            other_keys = [other_keys]
        if other_keys is None:
            other_keys = []

        info_keys: List[str] = ['train_instances']
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
            legacy=legacy)

    def merge(self, p: Tensor, r_p: Dict[str, Tensor], received_info: Dict, group: Dict) -> Any:
        train_instances = received_info['train_instances']
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        step = state['step']

        for key in group['pipe_keys']:
            if key in r_p:
                if key not in state:
                    state[key] = r_p[key]
                else:
                    state[key] = (state[key] * step +
                                  r_p[key] * train_instances) / (step + train_instances)
        state['step'] += train_instances

    def stack(self, p: Tensor, r_p: Dict[str, Tensor], received_info: Dict, **unused) -> Any:
        state = self.state[p]
        if 'received_params' not in state:
            state['received_params'] = []

        r_p["train_instances"] = received_info['train_instances']
        state['received_params'].append(r_p)

    def _merge_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]

        for key in group['pipe_keys']:
            if key in state:
                new_p = state[key]
                if key == "param":
                    if p.requires_grad:
                        p.grad.copy_(p - new_p)
                    else:
                        p.copy_(new_p)
                else:
                    ...

    def _stack_aggregate(self, p: Tensor, group: Dict):
        state = self.state[p]

        def aggregate(dl, k, t):
            l = 0
            for data in dl:
                a, b = data[k], data['train_instance']
                w = b / t
                p = a * w
                l += p
            return l

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
