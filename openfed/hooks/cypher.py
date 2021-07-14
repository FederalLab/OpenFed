from typing import Dict, Union

from openfed.common import Clone, peeper
from torch import Tensor

class Cypher(Clone):
    r"""Cypher: encrypt/decrypt data in pairs.
    The encrypt and decrypt functions will be called in two ends respectively.
    You can store the inner operation in the returned dictionary directly, but not 
    specify then as self.xxx=yyy.
    """

    def __init__(self):
        super().__init__()
        if peeper.api is not None:
            peeper.api.register_everything(self)

    def encrypt(self, key: Union[str, Tensor], value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package before transfer to the other end.
        """
        raise NotImplementedError(
            "You must implement the encrypt function for Cypher.")

    def decrypt(self, key: Union[str, Tensor], value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package received from the other end.
        """
        raise NotImplementedError(
            "You must implement the decrypt function for Cypher.")


class FormatCheck(Cypher):
    """Format Check.
    1. Convert `value` to {'param': value} if value is a Tensor.
    2. Align all other tensor in value to `param`'s device. 
    """

    def encrypt(self, key: Union[str, Tensor], value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert isinstance(key, Tensor)
        # Convert to dict
        if isinstance(value, Tensor):
            value = dict(param=value)
        assert isinstance(value, dict)

        # Align device
        for k, v in value.items():
            value[k] = v.cpu()
        return value

    def decrypt(self, key: Union[str, Tensor], value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert isinstance(key, Tensor)
        # Convert to dict
        if isinstance(value, Tensor):
            value = dict(param=value)
        assert isinstance(value, dict)

        # Align device
        for k, v in value.items():
            value[k] = v.to(key.device)
        return value

cyphers = [Cypher, FormatCheck]