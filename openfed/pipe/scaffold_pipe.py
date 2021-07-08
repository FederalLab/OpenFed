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


import torch

from .pipe import Pipe


class ScaffoldPipe(Pipe):
    """SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
    """

    def __init__(self, params, lr: float = None):
        """Scaffold need to run on both ft and backend.
        If lr is given, we will use the second way provided in the paper to update c.
        Otherwise, we will use the first way provided in the paper.
        If lr is not given, accumulate_gradient is needed.

        .. Example::
            >>> # lr is given.
            >>> scaffold = Scaffold(net.parameters(), lr=0.1, ft=True)
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
            >>> scaffold = Scaffold(net.parameters(), ft=True)
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
            >>> scaffold = Scaffold(net.parameters(), ft=False)
            >>> agg = Agg(net.parameters(), pipe_keys=scaffold.pack_key_list)
            >>> agg.agg()
            >>> scaffold.step()
            >>> optim.step()
            >>> deliver.pack(scaffold)
        """

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        self.add_pack_key('c_para')
        self.add_unpack_key('c_para')

        # Set initial c_para_i
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["c_para"] = torch.zeros_like(p)

    def _ft_step(self, closure=None, acg: bool = False):
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
                if acg:
                    if "init_p_g" not in state:
                        state["init_p_g"] = p.grad.clone().detach()
                    else:
                        state["init_p_g"].add_(p.grad)
                    continue

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

    def _bk_step(self, closure=None):
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

                # Update backend
                c_para = state["c_para"]
                if "c_para_i" not in state:
                    c_para_i = state["c_para_i"] = torch.zeros_like(p)
                else:
                    c_para_i = state["c_para_i"]

                c_para_i.add_(c_para)

                # copy c_para_i to c_para
                c_para.copy_(c_para_i)

        return loss

    def _ft_round(self):
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
                # Update ft
                if lr is None:
                    # Use the first way to update c_para
                    assert "init_p_g" in state, "You should accumulate init_p_g first!"
                    c_para_i.copy_(state["init_p_g"])
                else:
                    # Use the second way to update c_para
                    c_para_i.copy_(
                        c_para_i - state["c_para"] + 1 / (state["step"] * lr) * (state["init_p"] - p))
                state["c_para"].copy_(c_para_i - state["c_para"])

    def _bk_round(self):
        """Scaffold do a special round operation.
        Do not forget to call this when the round is finished.
        """

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                c_para_i = state["c_para_i"]
                state["c_para"].copy_(c_para_i - state["c_para"])

    def clear_buffer(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # delete init_p and step
                if 'init_p' in state:
                    del state["init_p"]
                if 'step' in state:
                    del state["step"]
