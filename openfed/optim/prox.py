# @Author            : FederalLab
# @Date              : 2021-09-25 16:53:43
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:53:43
# Copyright (c) FederalLab. All rights reserved.
from openfed.federated import collaborator
from .fed_optim import FederatedOptimizer


class ProxOptimizer(FederatedOptimizer):

    def __init__(self,
                 optim,
                 role: str = collaborator,
                 mu: float = 0.9,
                 max_acg_step: int = -1):
        if not 0.0 < mu < 1.0:
            raise ValueError(f'Invalid mu value: {mu}')

        self.mu = mu
        super(ProxOptimizer, self).__init__(optim, role, max_acg_step)

    def _collaborator_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'init_p' not in state:
                        init_p = state['init_p'] = p.clone().detach()
                    else:
                        init_p = state['init_p']
                    p.grad.add_(p - init_p, alpha=self.mu)

    def clear_state_dict(self):
        super().clear_state_dict(keys=['init_p'])
