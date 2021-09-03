# Copyright (c) FederalLab. All rights reserved.
import torch
import torch.nn.functional as F
from openfed.federated import follower

from .fed_optim import FederatedOptimizer


class ElasticOptimizer(FederatedOptimizer):
    def __init__(self,
                 optim,
                 role: str = follower,
                 momentum: float = 0.9,
                 max_acg_step=-1):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        self.momentum = momentum
        super(ElasticOptimizer, self).__init__(optim, role, max_acg_step)

    def acg(self, model, dataloader):
        model.train()
        device = next(model.parameters()).device

        for i, data in enumerate(dataloader):
            input, _ = data
            input = input.to(device)

            model.zero_grad()
            output = model(input)
            F.mse_loss(output, torch.zeros_like(output)).backward()
            self._acg_step()

            if self.max_acg_step > 0 and i > self.max_acg_step:
                break

    def _acg_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.abs()

                state = self.state[p]

                if 'importance' not in state:
                    state['importance'] = torch.zeros_like(p)
                state['importance'].mul_(self.momentum).add_(grad,
                                                             alpha=1 -
                                                             self.momentum)

    def clear_state_dict(self):
        return super().clear_state_dict(keys=['importance'])
