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


from threading import Lock
from typing import Any, List, Mapping, Tuple

class Array(object):
    """Make the class enable to iterate over a dict like list, 
    even though the dict has been motified while iteration.
    """

    # Array never modifies the _mapping.
    _mapping: Mapping[Any, Any]

    def __init__(self, mapping: Mapping[Any, Any]):
        assert isinstance(
            mapping, dict), "mapping must be a dict."
        self._mapping = mapping
        # Add a lock to make sure the list will not be modified during operation.
        self.lock = Lock()

        self.index = -1

    @property
    def mapping(self):
        assert self._mapping, f"{self} is not initialized by `Array.__init__`."
        return self._mapping

    @property
    def default_key(self) -> Any:
        return self[0][0]

    @property
    def keys(self) -> List[Any]:
        return list(self.mapping.keys())

    @property
    def default_value(self) -> Any:
        return self[0][1]

    @property
    def values(self) -> List[Any]:
        return list(self.mapping.values())

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Rectify index
        index = len(self) - 1 if index > len(self) else index

        self.current_keys = self.keys[index] if index > 0 else None
        self.current_values = self.values[index] if index > 0 else None

        return self.current_keys, self.current_values

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Any, Any]:
        self.index += 1
        if self.index >= len(self):
            self.index = -1
            raise StopIteration
        else:
            return self[self.index]

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, exc_type, exc_value, trace):
        self.lock.release()
