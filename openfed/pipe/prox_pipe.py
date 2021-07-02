import torch

from .base import Pipe


class ProxPipe(Pipe):
    """https://arxiv.org/pdf/1812.06127.pdf
    """

    def __init__(self, params, mu: float = 0.9):
        if not 0.0 < mu < 1.0:
            raise ValueError(f"Invalid mu value: {mu}")

        defaults = dict(mu=mu)
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
            mu = group['mu']
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if "init_p" not in state:
                        init_p = state["init_p"] = p.clone().detach()
                    else:
                        init_p = state["init_p"]
                    p.grad.add_(p-init_p, alpha=mu)

        return loss
