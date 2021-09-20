# Copyright (c) FederalLab. All rights reserved.
import torch
import torch.nn.functional as F
from typing import Callable, Union

from openfed.federated import collaborator
from .fed_optim import FederatedOptimizer


class ScaffoldOptimizer(FederatedOptimizer):

    def __init__(self,
                 optim,
                 role: str = collaborator,
                 max_acg_step: int = -1,
                 acg_loss_fn: Union[str, Callable] = 'cross_entropy'):
        super(ScaffoldOptimizer, self).__init__(optim, role, max_acg_step)
        if isinstance(acg_loss_fn, str):
            acg_loss_fn = getattr(F, acg_loss_fn)

        self.acg_loss_fn = acg_loss_fn
        self.init_c_para_flag = False

    def init_c_para(self):
        '''Call this function after glue operation.
        '''
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['c_para'] = torch.zeros_like(p)
        self.init_c_para_flag = True

    def acg(self, model, dataloader):
        '''Accumulate gradients for SCAFFOLD.
        Args:
            model: The model to test.
            dataloader: The data loader to iterate over.

        .. note::
            This function only be called if you do not specify the `lr` in
            `__init__` process.
        '''
        if self.init_c_para_flag is False:
            self.init_c_para()

        if self.acg_loss_fn is None:
            # It is not necessary to accumulate gradient with respect to the
            # second update rule described in the paper.
            return

        # accumulate gradient
        model.train()
        device = next(model.parameters()).device

        for i, data in enumerate(dataloader):
            input, target = data
            input, target = input.to(device), target.to(device)
            model.zero_grad()
            self.acg_loss_fn(model(input), target).backward()  # type: ignore
            self._acg_step()
            if self.max_acg_step > 0 and i > self.max_acg_step:
                break

    def _acg_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if 'init_p_g' not in state:
                    state['init_p_g'] = p.grad.clone().detach()
                    state['init_p_g_cnt'] = 1
                else:
                    g = (state['init_p_g'] * state['init_p_g_cnt'] +
                         p.grad) / (
                             state['init_p_g_cnt'] + 1)
                    state['init_p_g'].copy_(g)
                    state['init_p_g_cnt'] += 1

    def _collaborator_step(self, closure=None):
        '''Performs a single optimization step.

        Args:

            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            acg: accumulate gradient.
        '''
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if 'init_p' not in state:
                    state['init_p'] = p.clone().detach()
                if 'step' not in state:
                    state['step'] = 0

                state['step'] += 1
                # Modifed gradients
                if 'c_para_i' not in state:
                    c_para_i = state['c_para_i'] = torch.zeros_like(p)
                else:
                    c_para_i = state['c_para_i']
                # c_para will be loaded from agg/deliver automatically.
                assert 'c_para' in state, \
                    'c_para must be loaded from agg/deliver.'
                c_para = state['c_para']
                p.grad.add_(c_para - c_para_i, alpha=1)

        return loss

    def _aggregator_step(self, closure=None):
        '''Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        '''
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Update aggregator
                if 'c_para' not in state:
                    state['c_para'] = torch.zeros_like(p)
                c_para = state['c_para']
                if 'c_para_i' not in state:
                    c_para_i = state['c_para_i'] = torch.zeros_like(p)
                else:
                    c_para_i = state['c_para_i']

                c_para_i.add_(c_para)

                # copy c_para_i to c_para
                c_para.copy_(c_para_i)

        return loss

    def _collaborator_round(self):
        '''Scaffold do a special round operation.
        Do not forget to call this when the round is finished.
        '''

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                c_para_i = state['c_para_i']
                # Update collaborator
                if self.acg_loss_fn is not None:
                    # Use the first way to update c_para
                    assert 'init_p_g' in state, \
                        'You should accumulate init_p_g first!'
                    c_para_i.copy_(state['init_p_g'])
                else:
                    # Use the second way to update c_para
                    c_para_i.copy_(c_para_i - state['c_para'] + 1 /
                                   (state['step'] * group['lr']) *
                                   (state['init_p'] - p))
                if 'c_para' not in state:
                    state['c_para'] = c_para_i
                else:
                    state['c_para'].copy_(c_para_i - state['c_para'])

    def _aggregator_round(self):
        '''Scaffold do a special round operation.
        Do not forget to call this when the round is finished.
        '''

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                c_para_i = state['c_para_i']
                if 'c_para' not in state:
                    state['c_para'] = torch.zeros_like(p)
                state['c_para'].copy_(c_para_i - state['c_para'])

    def clear_state_dict(self):
        super().clear_state_dict(keys=['init_p_g', 'init_p_g_cnt', 'init_p'])
