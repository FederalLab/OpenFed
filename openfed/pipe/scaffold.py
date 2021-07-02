import torch
from .base import Pipe


class Scaffold(Pipe):
    """SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
    """

    def __init__(self, params, lr: float = None, frontend: bool = True):
        """Scaffold need to run on both frontend and backend.
        If lr is given, we will use the second way provided in the paper to update c.
        Otherwise, we will use the first way provided in the paper.
        If lr is not given, accumulate_gradient is needed.

        .. Example::
            >>> # lr is given.
            >>> scaffold = Scaffold(net.parameters(), lr=0.1, frontend=True)
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
            >>> scaffold = Scaffold(net.parameters(), frontend=True)
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
            >>> scaffold = Scaffold(net.parameters(), frontend=False)
            >>> aggregator = Aggregator(net.parameters(), pipe_keys=scaffold.pack_key_list)
            >>> aggregator.aggregate()
            >>> scaffold.step()
            >>> optim.step()
            >>> deliver.pack(scaffold)
        """

        self.add_pack_key('c_para')
        self.add_unpack_key('c_para')

        defaults = dict(lr=lr, frontend=frontend)
        super().__init__(params, defaults)

        # Set initial c_para_i
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["c_para"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None, accumulate_gradient: bool = True):
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
            lr = group['lr']
            frontend = group['frontend']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if accumulate_gradient:
                    if "init_p_g" not in state:
                        state["init_p_g"] = p.grad.clone().detach()
                    else:
                        state["init_p_g"].add_(p.grad)
                    continue
                if frontend:
                    if "init_p" not in state:
                        state["init_p"] = p.clone().detach()
                    if 'step' not in state:
                        state["step"] = 0
                    else:
                        state["step"] += 1
                    # Modifed gradients
                    if "c_para_i" not in state:
                        c_para_i = state["c_para_i"] = torch.zeros_like(p)
                    # c_para will be loaded from aggregator/deliver automatically.
                    assert "c_para" in state, "c_para must be loaded from aggregator/deliver."
                    c_para = state["c_para"]
                    p.grad.add_(c_para-c_para_i, alpha=1)
                else:
                    # Update backend
                    c_para = state["c_para"]
                    if "c_para_i" not in state:
                        c_para_i = state["c_para_i"] = torch.zeros_like(p)

                    c_para_i.add_(c_para)

                    # copy c_para_i to c_para
                    c_para.copy_(c_para_i)

        return loss

    def finish_round(self):
        """Scaffold do a special round operation.
        Do not forget to call this when the round is finished.
        """

        for group in self.param_groups:
            lr = group['lr']
            frontend = group['frontend']
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                c_para_i = state["c_para_i"]
                if frontend:
                    # Update frontend
                    if lr is None:
                        # Use the first way to update c_para
                        assert "init_p_g" in state, "You should accumulate init_p_g first!"
                        c_para_i.copy_(state["init_p_g"])
                    else:
                        # Use the second way to update c_para
                        c_para_i.copy_(
                            c_para_i - state["c_para"] + 1 / (state["step"] * lr) * (state["init_p"] - p))
                state["c_para"].copy_(c_para_i - state["c_para"])

                # del them
                del state["init_p"]
                del state["step"]
