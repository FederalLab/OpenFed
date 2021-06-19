from abc import abstractmethod
from typing import Dict

from torch import Tensor


class Cypher(object):
    r"""Cypher: encrypt/decrypt data in pairs.
    The encrypt and decrypt functions will be called in two ends respectively.
    You can store the inner operation in the returned dictionary directly, but not 
    specify then as self.xxx=yyy.
    """
    @abstractmethod
    def encrypt(self, key: str, value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package before transfer to the other end.
        """
        raise NotImplementedError(
            "You must implement the encrypt function for Cypher.")

    @abstractmethod
    def decrypt(self, key: str, value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package received from the other end.
        """
        raise NotImplementedError(
            "You must implement the decrypt function for Cypher.")


class FormotCheck(Cypher):
    """Make sure the value is a dict.
    """

    def encrypt(self, key: str, value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if isinstance(value, dict):
            return value
        elif isinstance(value, Tensor):
            return {"param": value}
        else:
            raise ValueError(f"{key}'s value is not desired.")

    def decrypt(self, key: str, value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if isinstance(value, dict):
            return value
        elif isinstance(value, Tensor):
            return {"param": value}
        else:
            raise ValueError(f"{key}'s value is not desired.")
