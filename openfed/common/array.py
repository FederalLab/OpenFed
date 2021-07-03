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


from typing import Any, List, Mapping
from threading import Lock


class Array(object):
    """Make the class enable to iterate over a dict like list, 
    even though the dict has been motified while iteration.
    """

    # Array will never modify the _default_mapping.
    _default_mapping: Mapping[Any, Any] = None
    _lock_on_mapping: Lock = None

    def __init__(self, default_mapping: Mapping[Any, Any], lock_on_mapping: Lock = None):
        assert default_mapping is not None,\
            "default_mapping can not be None, deliver an empty dict if you wanted."
        self._default_mapping = default_mapping
        self._lock_on_mapping = lock_on_mapping

        self.tmp_index = -1

    def _check_initialized_called(func):
        def wrapper(self, *args, **kwargs):
            assert self._default_mapping is not None,\
                "Call Array.__init__() to initialize Array first"
            return func(self, *args, **kwargs)
        return wrapper

    def _acquire_lock(func):
        def wrapper(self, *args, **kwargs):
            if self._lock_on_mapping:
                with self._lock_on_mapping:
                    output = func(self, *args, **kwargs)
            else:
                output = func(self, *args, **kwargs)
            return output
        return wrapper

    @property
    def default_keys(self):
        return self[0][0]

    @property
    def default_values(self):
        return self[0][1]

    @_check_initialized_called
    def __len__(self):
        return len(self._default_mapping)

    @_check_initialized_called
    @_acquire_lock  # Lock Here is Enough. DO NOT ADD IN OTHER FUNC! OTHERWISE WILL CAUSE DEAD LOCK!
    def __getitem__(self, index: int) -> List:
        if index >= len(self):
            index = len(self) - 1
        if index < 0:
            return None, None
        else:
            return list(self._default_mapping.keys())[index],\
                list(self._default_mapping.values())[index]

    def __iter__(self):
        return self

    def __next__(self):
        self.tmp_index += 1
        if self.tmp_index >= len(self):
            self.tmp_index = -1
            raise StopIteration
        else:
            return self[self.tmp_index]
