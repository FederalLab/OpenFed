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


from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, Union

from bidict import bidict
from openfed.common import ASYNC_OP, Hook, Package
from openfed.utils import openfed_class_fmt
from torch import Tensor
from torch._C._distributed_c10d import Work

from ..federated.functional import gather_object
from ..space import Country, ProcessGroup, World
from .cypher import Cypher, FormatCheck


class Delivery(Package, Hook):
    """Delivery: Include Tensor related communication function in a single class.
    """

    pg     : ProcessGroup
    country: Country
    world  : World

    key_tensor_bidict: bidict
    packages         : Dict[str, Dict[str, Tensor]]

    leader_rank  : int = 0
    follower_rank: int = 1

    def __init__(self) -> None:
        Hook.__init__(self)
        self.key_tensor_bidict = bidict()
        self.packages          = defaultdict(dict)
        self.register_cypher(FormatCheck())

    def register_cypher(self, cypher: Cypher) -> None:
        """Register a cypher to encrypt/decrypt the Tensor.
        """
        self.register_hook(cypher)

    def key_tensor_map(self, key: str, tensor: Tensor) -> None:
        """Add a new <key, tensor> pair to package.
        """
        if key in self.key_tensor_bidict or key == "param":
            raise KeyError(f"{key} already existed.")
        self.key_tensor_bidict[key] = tensor
        self.packages[key]["param"] = tensor

    def set_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Add a state_dict to package.
        """
        [self.key_tensor_map(k, v) for k, v in state_dict.items()]

    def reset_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Call reset() and set_state_dict() in a single step.
        """
        self.reset()
        self.set_state_dict(state_dict)

    def key_name(self, t: Tensor) -> str:
        """Return the string name for the given tensor t.
        """
        return self.key_tensor_bidict.inverse[t]

    def key_tensor(self, key: str) -> Tensor:
        """Return the tensor for the given key.
        """
        return self.key_tensor_bidict[key]

    def pack(self, key: Union[str, Tensor], rdict: Dict[str, Tensor]) -> None:
        """Update rdict to the key in package.
        """
        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        package = self.packages[key]

        package.update(rdict)

    def unpack(self, key: Union[str, Tensor], rdict: Dict[str, Any]) -> Dict[str, Tensor]:
        """Update rdict with the one saved in package.
        """
        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        package = self.packages[key]
        rdict   = {k: package[k] for k in rdict}

        return rdict

    @property
    def tensor_indexed_packages(self) -> Dict[Tensor, Dict[str, Tensor]]:
        """Return a Dict which indexed by Tensor.
        """
        return {self.key_tensor(k): v for k, v in self.packages.items()}

    def reset(self) -> None:
        """Reset key_tensor_bidict and packages.
        """
        self.key_tensor_bidict = bidict()
        self.packages = defaultdict(dict)

    def pull(self, auto_load_param: bool = True) -> Union[Dict[str, Dict[str, Tensor]], Tuple[Work, Callable]]:
        """Pull data from the other end. 
        After received data, Follower will load `param` to Tensor by an in-place operation automatically.
        You can specify :param:auto_load_param as ``False`` to disable it.
        """
        assert self.country._get_group_size(
            self.pg) == 2, "Delivery is only designed for group with size 2"

        received = [None, None]

        rank       = self.leader_rank if self.world.leader else self.follower_rank
        other_rank = self.follower_rank if self.world.leader else self.leader_rank

        def _op_after_gather(*args):
            r_packages: Dict = received[other_rank]

            # NOTE: decrypt data in the reverse order.
            for hook in self.hook_list[::-1]:
                r_packages = {k: hook.decrypt(self.key_tensor(k), v)
                              for k, v in r_packages.items()}

            # Follower will load `param` to Tensor by an in-place operation.
            if auto_load_param and self.world.follower:
                for k, v in r_packages.items():
                    if 'param' in v:
                        self.key_tensor_bidict[k].data.copy_(v['param'])
            self.packages = r_packages
            return r_packages

        returns = gather_object(
            None, received, 
            dst         = rank,
            group       = self.pg,
            async_op    = ASYNC_OP.is_async_op,
            country     = self.country,
            global_rank = False)

        if ASYNC_OP.is_async_op:
            handler, step_func = returns
            # lambda: before go into this layer's function, call step_func first.
            return handler, lambda: _op_after_gather(step_func())
        else:
            return _op_after_gather()

    def push(self) -> Union[None, Tuple[Work, Callable]]:
        """Push data to the other end.
        """
        assert self.country._get_group_size(
            self.pg) == 2, "Delivery is only designed for group with size 2"

        rank = self.follower_rank if self.world.leader else self.leader_rank

        # encrypt data
        for hook in self.hook_list:
            self.packages = {k: hook.encrypt(self.key_tensor(k), v)
                             for k, v in self.packages.items()}

        return gather_object(
            self.packages, None, 
            dst         = rank,
            group       = self.pg,
            async_op    = ASYNC_OP.is_async_op,
            country     = self.country,
            global_rank = False)

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name  = "Delivery",
            description = list(self.key_tensor_bidict.keys())
        )
