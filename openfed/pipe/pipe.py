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
from openfed.common import Wrapper
from torch.optim import Optimizer
from typing_extensions import final


class Pipe(Optimizer, Wrapper):
    """The basic class for federated pipe.

    Most federated optimizer just rectify the gradients according to
    some regulation, but not necessarily rewrite all the updating process.
    So, we device this Pipe class to do this.
    """

    def __init__(self, *args, **kwargs):
        Optimizer.__init__(self, *args, **kwargs)
        Wrapper.__init__(self)

    @torch.no_grad()
    @final
    def step(self, ft: bool = True, *args, **kwargs):
        return self._ft_step(*args, **kwargs) if ft else self._bk_step(*args, **kwargs)

    def _ft_step(self, *args, **kwargs):
        ...

    def _bk_step(self, *args, **kwargs):
        ...

    @torch.no_grad()
    @final
    def round(self, ft: bool = True):
        return self._ft_round() if ft else self._bk_round()

    def _ft_round(self):
        ...

    def _bk_round(self):
        ...

    def clear_buffer(self):
        """Only called at ft.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state[p]:
                    del self.state[p]
