import torch

from .base import Aux


class ElasticAux(Aux):
    r"""Paired with ElasticAggregator.

    Example:
        >>> elastic_aux = ElasticAux(net.parameters(), momentum=0.9)
        >>> while:
        >>>     elastic_aux.zero_grad()
        >>>     MSE(net(input), zeros).backward()
        >>>     elastic_aux.step()
        >>> elastic_aux.clear_state()

    """

    def __init__(self, params, momentum: float = 0.9):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

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
