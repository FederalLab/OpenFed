from typing import Any, List, Mapping


class Array(object):
    """Make the class enable to iterate over a dict like list, 
    even though the dict has been motified while iteration.
    """

    # Array will never modify the _default_mapping.
    _default_mapping: Mapping[Any, Any] = None

    def __init__(self, default_mapping: Mapping[Any, Any]):
        assert default_mapping is not None,\
            "default_mapping can not be None, deliver an empty dict if you wanted."
        self._default_mapping = default_mapping

        self.tmp_index = -1

    def _check_initialized_called(func):
        def wrapper(self, *args, **kwargs):
            assert self._default_mapping is not None,\
                "Call Array.__init__() to initialize Array first"
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
            return list(self._default_mapping.keys())[index],\
                list(self._default_mapping.values())[index]

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
