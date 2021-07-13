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


from typing import List, Union

from openfed.utils import convert_to_list


class Wrapper(object):
    """Provide some methods to wrap a class with support of Package.
    """
    _pack_key_list  : List[str]
    _unpack_key_list: List[str]

    @property
    def pack_key_list(self):
        if hasattr(self, '_pack_key_list'):
            return self._pack_key_list
        else:
            self._pack_key_list = list()
            return self._pack_key_list
    

    @property
    def unpack_key_list(self):
        if hasattr(self, '_unpack_key_list'):
            return self._unpack_key_list
        else:
            self._unpack_key_list = list()
            return self._unpack_key_list

    def add_pack_key(self, key: Union[str, List[str]]) -> None:
        key = convert_to_list(key)
        for k in key:
            if k in self.pack_key_list:
                raise KeyError(f"{k} is already registered.")
            self.pack_key_list.append(k)

    def add_unpack_key(self, key: Union[str, List[str]]) -> None:
        key = convert_to_list(key)
        for k in key:
            if k in self.unpack_key_list:
                raise KeyError(f"{k} is already registered.")
            self.unpack_key_list.append(k)
