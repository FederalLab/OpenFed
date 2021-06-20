from typing import Any, Dict, List, Union

from torch import Tensor
from torch.optim import Optimizer


class Aggregator():
    ...


def _check_state_keys(obj, keys: Union[str, List[str]], mode: str):
    keys = [keys] if isinstance(keys, str) else keys

    keys = keys if keys else getattr(obj, mode, None)

    return keys


class Package(object):
    """Provide function to pack and unpack state dict, like Optimizer...
    """

    def pack_state(self, obj: Union[Aggregator, Optimizer], keys: Union[str, List[str]] = None) -> None:
        """If keys is not given, we will try to load pack_key_list attribute.
        """
        keys = _check_state_keys(obj, keys, mode='pack_key_list')
        if keys:
            for group in obj.param_groups:
                for p in group["params"]:
                    state = obj.state[p]
                    rdict = {k: state[k] for k in keys}
                    self.pack(p, rdict)

    def unpack_state(self, obj: Union[Aggregator, Optimizer], keys: Union[str, List[str]] = None) -> None:
        """If keys is not given, we will try to load pack_key_list attribute.
        """
        keys = _check_state_keys(obj, keys, mode="unpack_key_list")
        if keys:
            for group in obj.param_groups:
                for p in group["params"]:
                    state = obj.state[p]
                    rdict = {k: None for k in keys}
                    rdict = self.unpack(p, rdict)
                    state.update(rdict)

    def pack(self, key: Union[str, Tensor], rdict: Dict[str, Tensor]) -> None:
        """Implement it in subclass if needed.
        """
        raise NotImplementedError

    def unpack(self, key: Union[str, Tensor], rdict: Dict[str, Any]) -> Dict[str, Tensor]:
        """Implement it in subclass if needed.
        """
        raise NotImplementedError
