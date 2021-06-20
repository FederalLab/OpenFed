import torch
from torch.optim import Optimizer
from typing import List
from ..common import Wrapper


class ElasticAux(Optimizer, Wrapper):
    r"""Paired with ElasticAggregator.

    Example:
        >>> elastic_aux = ElasticAux(net.parameters(), momentum=0.9)
        >>> while:
        >>>     elastic_aux.zero_grad()
        >>>     MSE(net(input), zeros).backward()
        >>>     elastic_aux.step()
        >>> elastic_aux.clear()

    """

    def __init__(self, params, momentum=0.9):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        self.add_pack_key('importance')

        defaults = dict(momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.abs()

                state = self.state[p]

                if 'importance' not in state:
                    state["importance"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                state["importance"].mul_(momentum).add_(grad, alpha=1-momentum)

        return loss

    def clear(self):
        """Clear accumulated importance weight.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'importance' in state:
                    del state['importance']
