# Copyright (c) FederalLab. All rights reserved.
from collections import OrderedDict
from typing import Any, Dict, List, Union, overload


class Attach(object):
    """Attach hooks to class.
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
        """Register hook to ``hook_dict``.
        """

    @overload
    def register_hook(self, func: Any):
        """Register hook to ``hook_list``.
        """

    def register_hook(self, *args):
        if len(args) == 1:
            func = args[0]
            if func in self.hook_list:
                raise RuntimeError(f"{func} is already registered.")

            index = -1
            for i, hook in enumerate(self.hook_list):
                if func.nice < hook.nice:
                    index = i
                    break
            if index == -1:
                self._hook_list.append(func)
            else:
                self._hook_list.insert(index, func)
        elif len(args) == 2:
            key, func = args
            if key in self.hook_dict:
                raise RuntimeError(f"{key} already registered.")
            new_hook_dict = OrderedDict()
            not_set = True
            for k, v in self.hook_dict.items():
                if func.nice < v.nice and not_set:
                    new_hook_dict[key] = func
                    not_set = False
                new_hook_dict[k] = v
            if not_set:
                new_hook_dict[key] = func
            self._hook_dict = new_hook_dict
        else:
            raise RuntimeError("Too many parameters.")

    def remove_hook(self, key: Union[str, Any]):
        """Remove ``key`` from ``hook_list`` or ``hook_dict``.
        Args: 
            key: If key is a string, remove it from ``hook_dict``,
                If key is a function, remove it from ``hook_list``.
        """
        if isinstance(key, str):
            if key not in self.hook_dict:
                raise RuntimeError(f"{key} is not registered.")
            del self._hook_dict[key]
        else:
            if key not in self.hook_list:
                raise RuntimeError(f"{key} is not registered")
            for i, func in enumerate(self._hook_list):
                if func == key:
                    del self._hook_list[i]
                    break
