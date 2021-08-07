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
from openfed.core import follower, leader
from openfed.common import Package
from openfed.utils import convert_to_list
from typing_extensions import final
import torch.nn.functional as F


class Penalizer(Package):
    """Penalizer is a special class that bound torch optimizer into 
    federated optimizer.

    Most of federated optimizer, actually, just rectifies the gradients
    according to specified regulations. Hence, we can split optimizer into
    two parts, the torch optimizer and penalizer. The latter one is responsible
    for the gradients rectifying but not update the param actually.
    """

    param_groups: dict  # assigned from optimizer
    state: dict  # assigned from optimizer

    def __init__(self,
                 role=follower,
                 pack_set: List[str] = None,
                 unpack_set: List[str] = None,
                 max_acg_step: int = -1):
        """
        Args:
            role: The role of current penalizer.
            pack_set: The inner state to upload, used for optimizer.
            unpack_set: The inner state to download, used for optimizer.
            max_acg_step: If max_acg_step < 0, we will iterate over the whole 
                dataset while call `acg()`.
        """
        self.role = role
        # Torch optimizer has no pack and unpack attributes,
        # So, we left this pack_key and unpack_key to compatible
        # with the torch optimizer. 
        self.add_pack_key(pack_set or [])
        self.add_unpack_key(unpack_set or [])
        
        self.max_acg_step = max_acg_step

    @property
    def leader(self):
        return self.role == leader

    @property
    def follower(self):
        return self.role == follower

    def acg(self, *args, **kwargs):
        """Used to compute some extra information before training.
        .. note::
            The parameters is not allowed to be modified in this function.
        """
        ...

    def _acg_step(self, *args, **kwargs):
        """inner access only.
        """
        ...

    @torch.no_grad()
    @final
    def step(self, closure=None):
        """Like `optimizer.step()`.
        """
        return self._follower_step(closure) if self.follower else self._leader_step(closure)

    def _follower_step(self, closure):
        """inner access only.
        """
        ...

    def _leader_step(self, closure):
        """inner access only.
        """
        ...

    @torch.no_grad()
    @final
    def round(self):
        """Called after each round is finished.
        """
        return self._follower_round() if self.follower else self._leader_round()

    def _follower_round(self):
        """inner access only.
        """
        ...

    def _leader_round(self):
        """inner access only.
        """
        ...

class PenalizerList(Penalizer):
    """You can chain different penalizers in a PenalizerList.
    It is very useful when you want to apply more than one
    penalizer on federated learning. However, by doing this, 
    you may not always gain performance improvements.
    PenalizerList provide the same feature as a single Penalizer.

    .. warn::
        Different with Penalizer, which will share all variables 
        and functions with Optimizer, PenalizerList will 
        only share the `param_groups` and `state` 
        variable from Optimizer. Pay attention to this.
    """
    def __init__(self, penalizer_list: List[Penalizer]):
        self.penalizer_list = penalizer_list

        assert len(set([p.role for p in self.penalizer_list])) == 1, 'The chained penalizer must have the same role.'

        # Merge pack and unpack set for each penalizer.
        for p in self.penalizer_list:
            self.pack_set.update(p.pack_set)
            self.unpack_set.update(p.unpack_set)

    @property
    def leader(self):
        return self.penalizer_list[0].leader
    
    @property
    def follower(self):
        return self.penalizer_list[0].follower

    def dynamic_build_penalizer(self, p: Penalizer):
        """Assign some variables and basic function to penalizer.
        .. note::
            Only param_groups, state will be assigned currently.
        """
        p.param_groups = self.param_groups
        p.state = self.state

    def acg(self, *args, **kwargs):
        for p in self.penalizer_list:
            self.dynamic_build_penalizer(p)
            p.acg(*args, **kwargs)

    def _follower_step(self, closure):
        for p in self.penalizer_list:
            self.dynamic_build_penalizer(p)
            p._follower_step(closure)

    def _leader_step(self, closure):
        for p in self.penalizer_list:
            self.dynamic_build_penalizer(p)
            p._leader_step(closure)

    def _follower_round(self):
        for p in self.penalizer_list:
            self.dynamic_build_penalizer(p)
            p._follower_round()
    
    def _leader_round(self):
        for p in self.penalizer_list:
            self.dynamic_build_penalizer(p)
            p._leader_round()

class ElasticPenalizer(Penalizer):
    r"""Elastic Penalizer is used for collecting some training statics 
    of client's data. Actually, it can be glued with other penalizers 
    to create new Pipe.

    .. note::
        Not any two penalizer can be glued, you have to make sure that
        the methods are not conflict. ElasticPenalizer only use the `acg_step`, 
        which make it couple well with other penalizers.
    """

    def __init__(self,
                 role: str,
                 momentum: float = 0.9,
                 pack_set: List[str] = None,
                 unpack_set: List[str] = None,
                 max_acg_step: int = -1):
        """
        Args:
            role: The role played.
            momentum: The momentum to accumulate importance weight.
            pack_set: ...
            unpack_set: ...
        """
        pack_set = convert_to_list(pack_set) or []
        pack_set.append('importance')

        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        # Do not record momentum to defaults.
        # It will make conflict with the momentum in other optimizer, 
        # such as SGD.
        self.momentum = momentum

        super().__init__(role, pack_set, unpack_set, max_acg_step)

    def acg(self, model, dataloader):
        """Accumulate gradients for elastic aggregation.
        Args:
            model: The model used to test.
            dataloader: The dataloader used to iterate over.

        The dataloader should return with [data, target] tuple.
        This is often used for classification task. 
        """
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

    def __init__(self, role: str, mu: float = 0.9, pack_set: List[str] = None, unpack_set: List[str] = None):
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

    def __init__(self, 
        role      : str,
        lr        : float = None,
        pack_set  : List[str] = None,
        unpack_set: List[str] = None, 
        max_acg_step: int = -1, 
        acg_loss_fn: Union[Callable, str] = 'mse_loss'): 
        """Scaffold needs to run on both leader and follower.
        If lr is given, we will use the second way provided in the paper to update c.
        Otherwise, we will use the first way provided in the paper.
        If lr is not given, acg_step is needed.

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

        self.lr = lr
        if isinstance(acg_loss_fn, str):
            acg_loss_fn = getattr(F, acg_loss_fn)

        self.acg_loss_fn = acg_loss_fn

    def init_c_para(self):
        """Call this function after glue operation.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["c_para"] = torch.zeros_like(p)

    def acg(self, model, dataloader):
        """Accumulate gradients for SCAFFOLD.
        Args:
            model: The model to test.
            dataloader: The data loader to iterate over.

        .. note:: 
            This function only be called if you do not specify the `lr` in 
            `__init__` process.
        """
        if self.lr is not None:
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
            self.acg_loss_fn(model(input), target).backward() # type: ignore
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
