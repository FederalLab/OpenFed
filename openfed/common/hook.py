from collections import OrderedDict
from typing import Callable, Dict, List, overload

from .logger import log_error_info


class Hook(object):
    """为那些具有钩子函数的类，提供一些固定的方法。
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
        """指定key和value，会将他们注册到hook_dict里面去
        """

    @overload
    def register_hook(self, func: Callable):
        """单独指定func，会将他们注册到hook_list里面去
        """

    def register_hook(self, **kwargs):
        key = kwargs.get('key', None)
        if key:
            if key in self.hook_dict:
                msg = "Key '%s' already registered" % key
                log_error_info(msg)
                raise KeyError(msg)
            self.hook_dict[key] = kwargs['func']
        else:
            func = kwargs['func']
            if func in self.hook_list:
                msg = "Key '%s' already registered" % key
                log_error_info(msg)
                raise KeyError(msg)
            self.hook_list.append(func)
