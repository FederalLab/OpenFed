import torch
from openfed.common import Wrapper
from torch.optim import Optimizer


class Pipe(Optimizer, Wrapper):
    """The basic class for federated optimizer pipeiliary.

    Most federated optimizer just rectify the gradients according to
    some regulation, but not necessarily rewrite all the updating process.
    So, we device this Pipe class to do this.
    """

    @torch.no_grad()
    def finish_round(self):
        """Update self state after train a round. (Mostly clear the state directly.)
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state[p]:
                    del self.state[p]
