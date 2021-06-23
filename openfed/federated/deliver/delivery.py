from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, Union

from bidict import bidict
from openfed.common import Hook, Package
from openfed.common.vars import ASYNC_OP
from openfed.federated.country import Country, ProcessGroup
from openfed.federated.deliver.functional import Cypher, FormotCheck
from openfed.federated.functional import gather_object
from openfed.federated.world import World
from openfed.utils import openfed_class_fmt
from torch import Tensor
from torch._C._distributed_c10d import Work


class Delivery(Package, Hook):
    """Delivery: Include Tensor related communication function in a single class.
    """

    pg: ProcessGroup
    country: Country
    world: World

    key_tensor_bidict: bidict
    packages: Dict[str, Dict[str, Tensor]]

    def __init__(self) -> None:
        self.key_tensor_bidict = bidict()
        self.packages = defaultdict(dict)
        self.register_cypher(FormotCheck())

    @property
    def king_rank(self) -> int:
        return 0

    @property
    def queen_rank(self) -> int:
        return 1

    def register_cypher(self, cypher: Cypher) -> None:
        """Register a cypher to encrypt/decrypt the Tensor.
        """
        Hook.register_hook(self, func=cypher)

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
        for k, v in state_dict.items():
            self.key_tensor_map(k, v)

    def key_name(self, t: Tensor) -> str:
        """Return the string name for the given tensor t.
        """
        return self.key_tensor_bidict.inverse[t]

    def key_tensor(self, key: str) -> Tensor:
        """Return the tensor for the given key.
        """
        return self.key_tensor_bidict.get(key)

    def pack(self, key: Union[str, Tensor], rdict: Dict[str, Tensor]) -> None:
        """Update rdict to the key in package.
        """
        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        package = self.packages.get(key)

        package.update(rdict)

    def unpack(self, key: Union[str, Tensor], rdict: Dict[str, Any]) -> Dict[str, Tensor]:
        """Update rdict with the one saved in package.
        """
        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        package = self.packages.get(key)
        rdict = {k: package[k] for k in rdict}

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
        After received data, Queen will load `param` to Tensor by an in-palce operation automatically.
        You can specify :param:auto_load_param as ``False`` to disable it.
        """
        assert self.country._get_group_size(
            self.pg) == 2, "Delivery is only designed for group with size 2"

        received = [None, None]

        rank = self.king_rank if self.world.king else self.queen_rank
        other_rank = self.queen_rank if self.world.king else self.king_rank

        def _op_after_gather(*args):
            r_packages = received[other_rank]

            # NOTE: decrypt data in the reverse order.
            for hook in self.hook_list[::-1]:
                r_packages = {k: hook.decrypt(k, v)
                              for k, v in r_packages.items()}

            # Queen will load `param` to Tensor by an in-place operation.
            if auto_load_param and self.world.queen:
                for k, v in r_packages.items():
                    if 'param' in v:
                        self.key_tensor_bidict[k].data.copy_(v['param'])
            self.packages = r_packages
            return r_packages

        returns = gather_object(
            None, received, dst=rank, group=self.pg,
            async_op=ASYNC_OP.is_async_op,
            country=self.country)

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

        rank = self.queen_rank if self.world.king else self.king_rank

        # encrypt data
        for hook in self.hook_list:
            self.packages = {k: hook.encrypt(k, v)
                             for k, v in self.packages.items()}

        return gather_object(
            self.packages, None, dst=rank,
            group=self.pg, async_op=ASYNC_OP.is_async_op, country=self.country)

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Delivery",
            description=str(list(self.key_tensor_bidict.keys()))
        )
