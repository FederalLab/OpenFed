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
from typing import Callable, Dict, List, overload


class Hook(object):
    """Provide some functions to register and manage hooks.
    """

    _hook_dict: Dict[str, Callable] = None
    _hook_list: List[Callable] = None

    @property
    def hook_list(self):
        if self._hook_list is None:
            self._hook_list = []
        return self._hook_list

    @property
    def hook_dict(self):
        if self._hook_dict is None:
            self._hook_dict = OrderedDict()
        return self._hook_dict

    @overload
    def register_hook(self, key: str, func: Callable):
        """Add hook to _hook_dict.
        """

    @overload
    def register_hook(self, func: Callable):
        """Add hook to _hook_list
        """

    def register_hook(self, **kwargs):
        key = kwargs.get('key', None)
        if key:
            if key in self.hook_dict:
                msg = "Key '%s' already registered" % key
                raise KeyError(msg)
            self.hook_dict[key] = kwargs['func']
        else:
            func = kwargs['func']
            if func in self.hook_list:
                msg = "Func '%s' already registered" % func
                raise KeyError(msg)
            self.hook_list.append(func)
