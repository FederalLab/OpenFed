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
                msg = "Key '%s' already registered" % key
                raise KeyError(msg)
            self.hook_list.append(func)
