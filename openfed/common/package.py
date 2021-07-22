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


from typing import Any, Dict, List, Set, Union

from openfed.utils import convert_to_list
from torch import Tensor
import torch

class Package(object):
    """Pack Optimizer, Penalizer and Container state to state dict. 
    Unpack received state dict to Optimizer, Penalizer and Container.
    """
    _pack_set   : Set[str]
    _unpack_set : Set[str]
    param_groups: List[Dict]
    state       : Dict

    def pack_state(self, obj, keys=None) -> None:
        """Pack ``obj.state_dict`` to ``self.state_dict``.
        Args:
            obj (Penalizer, Pipe, Container): The object contains state dict to fetch data.
            keys: The extra keys want to pack to Package.
        """
        keys     = convert_to_list(keys) or []
        all_keys = obj.pack_set
        for key in keys:
            if key not in all_keys:
                all_keys.append(key)

        for group in obj.param_groups:
            for p in group["params"]:
                state = obj.state[p]
                # Sometimes, not all params will contains all the keys,
                # Here, we just return the contains one.
                rdict = {k: state[k] for k in all_keys if k in state}
                self.pack(p, rdict)

    def unpack_state(self, obj, keys=None) -> None:
        """Unpack ``self.state_dict`` to ``obj.state_dict``. 
        This is the reverse process of `self.pack_state`.
        Args:
            obj (Penalizer, Pipe, Container): The object contains state dict to fill data.
            keys: The extra keys want to unpack to obj.
        """
        keys     = convert_to_list(keys) or []
        all_keys = obj.unpack_set
        for key in keys:
            if key not in all_keys:
                all_keys.append(key)

        for group in obj.param_groups:
            for p in group["params"]:
                state = obj.state[p]
                # Fill `None` to all keys.
                rdict = {k: None for k in keys}
                state.update(self.unpack(p, rdict))

    def pack(self,
             key  : Union[str, Tensor],
             rdict: Dict[str, Any]) -> None: 
        """Pack ``rdict`` to ``self.state_dict[key]``.
        """
        state = self.state[key]
        state.update(rdict)

    def unpack(self,
               key  : Union[str, Tensor],
               rdict: Dict[str, Any]) -> Dict[str, Any]: 
        """Unpack ``self.state_dict[key]`` to ``rdict``.
        """
        state = self.state[key]
        rdict.update({k: state[k] for k in rdict})
        return rdict

    @property
    def pack_set(self):
        if not hasattr(self, '_pack_set'):
            self._pack_set = set()
        return self._pack_set

    @property
    def unpack_set(self):
        if not hasattr(self, '_unpack_set'):
            self._unpack_set = set()
        return self._unpack_set

    def add_pack_key(self, keys: Union[str, List[str]]):
        """Add a new key to ``pack_set``.
        """
        [self.pack_set.add(k) for k in convert_to_list(keys)]

    def add_unpack_key(self, keys: Union[str, List[str]]):
        """Add a new key to ``unpack_set``.
        """
        [self.unpack_set.add(k) for k in convert_to_list(keys)]

    def clear_buffer(self, keep_keys: List[str] = None):
        """Clear the key-value in ``state_dict``.
        Args:
            keep_keys: The list of keys which will not be cleared.

        .. note::
            If ``keep_keys`` in ``self.param_groups``, the passed one
        will be discarded.
        """
        keep_keys = convert_to_list(keep_keys) or []

        for group in self.param_groups:
            if 'keep_keys' in group:
                keys = group['keep_keys'] or keep_keys
            else:
                keys = keep_keys
            for p in group["params"]:
                if p in self.state[p]:
                    state = self.state[p]
                    del_keys = [key for key in state.keys() if key not in keys]
                    for key in del_keys:
                        del state[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()