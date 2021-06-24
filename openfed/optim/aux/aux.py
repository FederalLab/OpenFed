import torch
from openfed.common.wrapper import Wrapper
from torch.optim import Optimizer


class Aux(Optimizer, Wrapper):
    """The basic class for federated optimizer auxiliary.

    Most federated optimizer just rectify the gradients according to
    some regulation, but not necessarily rewrite all the updating process.
    So, we device this Aux class to do this.
    """

    @torch.no_grad()
    def clear_state(self):
        """Clear the state cached in this round, then it can be reused.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state[p]:
                    del self.state[p]
