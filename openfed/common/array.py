from typing import Any, List, Mapping


class Array(object):
    """
    提供一些方法，使得一个类能够实现一些数组的基本功能，以及高级功能。
    提供了迭代过程中对象被修改而影响程序运行的能力。
    注意：当迭代对象被修改以后，可能会乱序迭代。你不应当期待它会提供有序的迭代的能力。虽然有时候他看起来可以。
    在需要对一些共享的全局mapping变量进行迭代时，显得非常有用。
    """

    # 下划线开头，这个类本身不会对mapping做出任何修改
    _default_mapping: Mapping[Any, Any] = None

    def __init__(self, default_mapping: Mapping[Any, Any]):
        assert default_mapping is not None, "default_mapping can not be None, deliver an empty dict if you wanted."
        self._default_mapping = default_mapping

        self.tmp_index = -1

    def _check_initialized_called(func):
        def wrapper(self, *args, **kwargs):
            assert self._default_mapping is not None, "Call Array.__init__ to initialized Array first"
            return func(self, *args, **kwargs)
        return wrapper

    @property
    @_check_initialized_called
    def default_keys(self):
        return self[0][0]

    @property
    @_check_initialized_called
    def default_values(self):
        return self[0][1]

    @_check_initialized_called
    def __len__(self):
        return len(self._default_mapping)

    @_check_initialized_called
    def __getitem__(self, index: int) -> List:
        if index >= len(self):
            index = len(self) - 1
        if index < 0:
            return None, None
        else:
            return list(self._default_mapping.keys())[index], list(self._default_mapping.values())[index]

    @_check_initialized_called
    def __iter__(self):
        return self

    @_check_initialized_called
    def __next__(self):
        self.tmp_index += 1
        if self.tmp_index >= len(self):
            self.tmp_index = -1
            raise StopIteration
        else:
            return self[self.tmp_index]
