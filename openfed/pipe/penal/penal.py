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
from openfed.common import Wrapper
from typing_extensions import final


class Penalizer(Wrapper):
    """The basic class for federated pipe.

    Most federated optimizer just rectify the gradients according to
    some regulation, but not necessarily rewrite all the updating process.
    So, we device this Pipe class to do this.
    """

    param_groups: dict  # assigned from optimizer
    state: dict  # assigned from optimizer

    def __init__(self,
                 ft: bool = True,
                 pack_key_list: List[str] = None,
                 unpack_key_list: List[str] = None):
        self.ft = ft
        if pack_key_list is not None:
            self.add_pack_key(pack_key_list)
        if unpack_key_list is not None:
            self.add_unpack_key(unpack_key_list)

    @torch.no_grad()
    @final
    def step(self, closure):
        return self._ft_step(closure) if self.ft else self._bk_step(closure)

    def _ft_step(self, closure):
        ...

    def _bk_step(self, closure):
        ...

    @torch.no_grad()
    @final
    def round(self):
        return self._ft_round() if self.ft else self._bk_round()

    def _ft_round(self):
        ...

    def _bk_round(self):
        ...

    def clear_buffer(self, keep_keys: List[str] = None):
        """Clear state buffers.
        Args:
            keep_keys: if not specified, we will directly remove all buffers.
                Otherwise, the key in keep_keys will be kept.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state[p]:
                    if keep_keys is None:
                        del self.state[p]
                    else:
                        for k in self.state[p].keys():
                            if k not in keep_keys:
                                del self.state[p][k]
