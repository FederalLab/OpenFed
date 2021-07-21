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


from typing import List

import torch
from openfed.core import ROLE, follower, leader
from openfed.common import Package
from openfed.utils import convert_to_list
from typing_extensions import final


class Penalizer(Package):
    """The basic class for federated pipe.

    Most federated optimizer just rectify the gradients according to
    some regulation, but not necessarily rewrite all the updating process.
    So, we device this Pipe class to do this.
    """

    param_groups: dict  # assigned from optimizer
    state: dict  # assigned from optimizer

    def __init__(self,
                 role = follower,
                 pack_set: List[str] = None,
                 unpack_set: List[str] = None):
        self.role = role
        if pack_set is not None:
            self.add_pack_key(pack_set)
        if unpack_set is not None:
            self.add_unpack_key(unpack_set)

    @torch.no_grad()
    @final
    def step(self, closure=None):
        return self._follower_step(closure) if self.role == follower else self._leader_step(closure)

    def _follower_step(self, closure):
        ...

    def _leader_step(self, closure):
        ...

    @torch.no_grad()
    @final
    def round(self):
        return self._follower_round() if self.role == follower else self._leader_round()

    def _follower_round(self):
        ...

    def _leader_round(self):
        ...

class ElasticPenalizer(Penalizer):
    r"""Paired with ElasticAgg.

    Example:
        >>> elastic_pipe = ElasticPipe(net.parameters(), momentum=0.9)
        >>> while:
        >>>     elastic_pipe.zero_grad()
        >>>     MSE(net(input), zeros).backward()
        >>>     elastic_pipe.step()
        >>> elastic_pipe.clear_state()

    """

    def __init__(self,
                 role: ROLE,
                 momentum: float = 0.9,
                 pack_set: List[str] = None,
                 unpack_set: List[str] = None):
        pack_set = convert_to_list(pack_set)
        unpack_set = convert_to_list(unpack_set)
        if pack_set is None:
            pack_set = ['importance']
        else:
            pack_set.append('importance')

        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        self.momentum = momentum

        super().__init__(role, pack_set, unpack_set)

    def acg_step(self):
        """Performs a single accumulate gradient step.
        """
        for group in self.param_groups:
            momentum = self.momentum
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.abs()
                state = self.state[p]
                if 'importance' not in state:
                    state["importance"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                state["importance"].mul_(momentum).add_(grad, alpha=1-momentum)


class ProxPenalizer(Penalizer):
    """https://arxiv.org/pdf/1812.06127.pdf
    """

    def __init__(self, role: ROLE, mu: float = 0.9, pack_set: List[str] = None, unpack_set: List[str] = None):
        if not 0.0 < mu < 1.0:
            raise ValueError(f"Invalid mu value: {mu}")

        super().__init__(role, pack_set, unpack_set)
        self.mu = mu

    def _follower_step(self, closure=None):
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


class ScaffoldPenalizer(Penalizer):
    """SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
    """

    def __init__(self, role: ROLE, lr: float = None, pack_set: List[str] = None, unpack_set: List[str] = None):
        """Scaffold need to run on both leader and follower.
        If lr is given, we will use the second way provided in the paper to update c.
        Otherwise, we will use the first way provided in the paper.
        If lr is not given, accumulate_gradient is needed.

        .. Example::
            >>> # lr is given.
            >>> scaffold = Scaffold(net.parameters(), lr=0.1, role=follower)
            >>> openfed_api.unpack(scaffold)
            >>> for data in dataloader:
            >>>     optim.zero_grad()
            >>>     net(data).backward()
            >>>     scaffold.step()
            >>>     optim.step()
            >>> # round end
            >>> scaffold.finish_round()
            >>> openfed_api.pack(scaffold)

            >>> # lr not given
            >>> scaffold = Scaffold(net.parameters(), role=follower)
            >>> openfed_api.unpack(scaffold)
            >>> # Accumulate Gradient Stage
            >>> for data in dataloader:
            >>>     net(data).backward()
            >>> scaffold.step(accumulate_gradient=True)
            >>> # Train
            >>> for data in dataloader:
            >>>     optim.zero_grad()
            >>>     net(data).backward()
            >>>     scaffold.step()
            >>>     optim.step()
            >>> # round end
            >>> scaffold.finish_round()
            >>> openfed_api.pack(scaffold)

            >>> # Backend
            >>> scaffold = Scaffold(net.parameters(), role=leader)
            >>> agg = Agg(net.parameters(), pipe_keys=scaffold.pack_set)
            >>> agg.agg()
            >>> scaffold.step()
            >>> optim.step()
            >>> deliver.pack(scaffold)
        """
        pack_set = convert_to_list(pack_set)
        unpack_set = convert_to_list(unpack_set)
        if pack_set is not None:
            pack_set.append('c_para')
        else:
            pack_set = ['c_para']

        if unpack_set is not None:
            unpack_set.append('c_para')
        else:
            unpack_set = ['c_para']
        super().__init__(role, pack_set, unpack_set)

        self.lr = lr

    def init_c_para(self):
        """Call this function after glue operation.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["c_para"] = torch.zeros_like(p)

    def acg_step(self):
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
                else:
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
            lr = group['lr']
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                c_para_i = state["c_para_i"]
                # Update follower
                if lr is None:
                    # Use the first way to update c_para
                    assert "init_p_g" in state, "You should accumulate init_p_g first!"
                    c_para_i.copy_(state["init_p_g"])
                else:
                    # Use the second way to update c_para
                    c_para_i.copy_(
                        c_para_i - state["c_para"] + 1 / (state["step"] * lr) * (state["init_p"] - p))
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


penalizers = [
    Penalizer, ElasticPenalizer, ProxPenalizer, ScaffoldPenalizer
]
