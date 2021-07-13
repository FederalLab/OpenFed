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


from collections import OrderedDict
from typing import Any, Dict, List, Union, overload


class Hook(object):
    """Provide some functions to register and manage hooks.
    """

    _hook_dict: Dict[str, Any]
    _hook_list: List[Any]

    @property
    def hook_list(self):
        if not hasattr(self, '_hook_list'):
            self._hook_list = list()
        return self._hook_list

    @property
    def hook_dict(self):
        if not hasattr(self, '_hook_dict'):
            self._hook_dict = OrderedDict()
        return self._hook_dict

    @overload
    def register_hook(self, key: str, func: Any):
        """Register a func with key to hook dictionary.
        """

    @overload
    def register_hook(self, func: Any):
        """Register a func to hook list.
        """

    def register_hook(self, *args):
        if len(args) == 1:
            func = args[0]
            if func in self.hook_list:
                raise KeyError(f"{func} is already registered.")
            self._hook_list.append(func)
        elif len(args) == 2:
            key, func = args
            if key in self.hook_dict:
                raise KeyError(f"Key {key} already registered.")
            self.hook_dict[key] = func
        else:
            raise RuntimeError("Too many parameters.")

    def remove_hook(self, key: Union[str, Any]):
        """
        Args: 
            key: if key is str, remove it from hook dict.
                else, remove it from hook list.
        """
        if isinstance(key, str):
            if key in self._hook_dict:
                del self._hook_dict[key]
        else:
            for i, func in enumerate(self._hook_list):
                if func == key:
                    del self._hook_list[i]
                    break
