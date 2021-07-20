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


from typing import Any, Dict, List, Union

from openfed.utils import convert_to_list
from torch import Tensor


def _check_state_keys(obj, keys: Union[None, str, List[str]], mode: str):
    keys = convert_to_list(keys)

    return keys if keys else getattr(obj, mode, None)


class Package(object):
    """Define some unified functions to package and unpackage the class's state dictionary, such as Optimizer, Agg and Pipe.
    """
    _pack_key_list: List[str]
    _unpack_key_list: List[str]
    param_groups: Any
    state: Any

    def pack_state(self, obj, keys: Union[None, str, List[str]] = None) -> None:
        """
        Args:
            keys: if keys are not given, we will try to load the `pack_key_list` attribute of obj.
        """
        keys = _check_state_keys(obj, keys, mode='pack_key_list')
        if keys:
            for group in obj.param_groups:
                for p in group["params"]:
                    state = obj.state[p]
                    rdict = {k: state[k] for k in keys if k in state}
                    self.pack(p, rdict)

    def unpack_state(self, obj, keys: Union[None, str, List[str]] = None) -> None:
        """
        Args:
            keys: if keys are not given, we will try to load the `unpack_key_list` attribute of obj.
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

    @property
    def pack_key_list(self):
        if not hasattr(self, '_pack_key_list'):
            self._pack_key_list = list()
        return self._pack_key_list

    @property
    def unpack_key_list(self):
        if not hasattr(self, '_unpack_key_list'):
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

    def clear_buffer(self, keep_keys: List[str] = None):
        """Clear state buffers.
        Args:
            keep_keys: if not specified, we will directly remove all buffers.
                Otherwise, the key in keep_keys will be kept.
        """
        keep_keys = convert_to_list(keep_keys)

        for group in self.param_groups:
            if 'keep_keys' in group:
                if keep_keys is None:
                    keys = group['keep_keys']
                elif group['keep_keys'] is None:
                    keys = keep_keys
                else:
                    keys = keep_keys + group['keep_keys']
            else:
                keys = keep_keys

            for p in group["params"]:
                if p in self.state[p]:
                    if keys is None:
                        del self.state[p]
                    else:
                        for k in self.state[p].keys():
                            if k not in keys:
                                del self.state[p][k]