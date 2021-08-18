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
from openfed.core import follower, leader
from openfed.common import Package
from typing_extensions import final


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
        return self._follower_step(
            closure) if self.follower else self._leader_step(closure)

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
        return self._follower_round() if self.follower else self._leader_round(
        )

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

        assert len(set([
            p.role for p in self.penalizer_list
        ])) == 1, 'The chained penalizer must have the same role.'

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
