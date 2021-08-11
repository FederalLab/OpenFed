# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import List, Callable, Union

import torch
from openfed.utils import convert_to_list
import torch.nn.functional as F

from .penal import Penalizer


class ScaffoldPenalizer(Penalizer):
    """SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
    """

    def __init__(self,
                 role: str,
                 pack_set: List[str] = None,
                 unpack_set: List[str] = None,
                 max_acg_step: int = -1,
                 acg_loss_fn: Union[Callable, str] = 'cross_entropy'):
        """Scaffold needs to run on both leader and follower.
        If acg_loss_fn is not None, we will use the second way described in the
        paper to do the acg step.

        Args:
            acg_loss_fn: The callable loss function used to calculate acg operation.
                At most of time, it should be the consit with the loss function used 
                in the task itself, such as BCE, MSE...,

        """
        pack_set = convert_to_list(pack_set) or []
        unpack_set = convert_to_list(unpack_set) or []
        pack_set.append('c_para')
        unpack_set.append('c_para')

        super().__init__(role, pack_set, unpack_set, max_acg_step)

        if isinstance(acg_loss_fn, str):
            acg_loss_fn = getattr(F, acg_loss_fn)

        self.acg_loss_fn = acg_loss_fn
        self.init_c_para_flag = False

    def init_c_para(self):
        """Call this function after glue operation.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["c_para"] = torch.zeros_like(p)
        self.init_c_para_flag = True

    def acg(self, model, dataloader):
        """Accumulate gradients for SCAFFOLD.
        Args:
            model: The model to test.
            dataloader: The data loader to iterate over.

        .. note:: 
            This function only be called if you do not specify the `lr` in 
            `__init__` process.
        """
        if self.init_c_para_flag == False:
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

                if "init_p_g" not in state:
                    state["init_p_g"] = p.grad.clone().detach()
                    state["init_p_g_cnt"] = 1
                else:
                    g = (state["init_p_g"] * state["init_p_g_cnt"] +
                         p.grad) / (state["init_p_g_cnt"] + 1)
                    state["init_p_g"].copy_(g)
                    state['init_p_g_cnt'] += 1

    def _follower_step(self, closure=None):
        """Performs a single optimization step.

        Args:

            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            acg: accumulate gradient.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if "init_p" not in state:
                    state["init_p"] = p.clone().detach()
                if 'step' not in state:
                    state["step"] = 0

                state["step"] += 1
                # Modifed gradients
                if "c_para_i" not in state:
                    c_para_i = state["c_para_i"] = torch.zeros_like(p)
                else:
                    c_para_i = state["c_para_i"]
                # c_para will be loaded from agg/deliver automatically.
                assert "c_para" in state, "c_para must be loaded from agg/deliver."
                c_para = state["c_para"]
                p.grad.add_(c_para-c_para_i, alpha=1)

        return loss

    def _leader_step(self, closure=None):
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
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Update leader
                if 'c_para' not in state:
                    state['c_para'] = torch.zeros_like(p)
                c_para = state["c_para"]
                if "c_para_i" not in state:
                    c_para_i = state["c_para_i"] = torch.zeros_like(p)
                else:
                    c_para_i = state["c_para_i"]

                c_para_i.add_(c_para)

                # copy c_para_i to c_para
                c_para.copy_(c_para_i)

        return loss

    def _follower_round(self):
        """Scaffold do a special round operation.
        Do not forget to call this when the round is finished.
        """

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                c_para_i = state["c_para_i"]
                # Update follower
                if self.acg_loss_fn is not None:
                    # Use the first way to update c_para
                    assert "init_p_g" in state, "You should accumulate init_p_g first!"
                    c_para_i.copy_(state["init_p_g"])
                else:
                    # Use the second way to update c_para
                    c_para_i.copy_(
                        c_para_i - state["c_para"] + 1 / (state["step"] * group['lr']) * (state["init_p"] - p))
                if 'c_para' not in state:
                    state["c_para"] = c_para_i
                else:
                    state["c_para"].copy_(c_para_i - state["c_para"])

    def _leader_round(self):
        """Scaffold do a special round operation.
        Do not forget to call this when the round is finished.
        """

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                c_para_i = state["c_para_i"]
                if 'c_para' not in state:
                    state['c_para'] = torch.zeros_like(p)
                state["c_para"].copy_(c_para_i - state["c_para"])

    def clear_buffer(self):
        super().clear_buffer(keep_keys=['c_para_i', 'c_para'])
