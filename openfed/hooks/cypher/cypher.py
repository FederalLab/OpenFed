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

from typing import Dict, Union

from torch import Tensor

from ..hooks import Hooks


class Cypher(Hooks):
    r"""Cypher: encrypt/decrypt data in pairs.
    The encrypt and decrypt functions will be called in two ends respectively.
    You can store the inner operation in the returned dictionary directly, but not 
    specify then as self.xxx=yyy.
    """
    def encrypt(self, key: Union[str, Tensor],
                value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package before transfer to the other end.
        """
        raise NotImplementedError(
            "You must implement the encrypt function for Cypher.")

    def decrypt(self, key: Union[str, Tensor],
                value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package received from the other end.
        """
        raise NotImplementedError(
            "You must implement the decrypt function for Cypher.")