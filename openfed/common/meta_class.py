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
from typing import Any, Tuple

from openfed.utils import openfed_class_fmt, tablist


class AttrDict(dict):
    """Attribute Dictionary. You can access the items like attributes.

    .. Example::
        >>> attr_dict = AttrDict(openfed='openfed')
        >>> attr_dict.openfed
        'openfed'
        >>> attr_dict
        {'openfed': 'openfed'}
        >>> attr_dict.version = 1.0
        >>> attr_dict
        {'openfed': 'openfed', 'version': 1.0}
        >>> attr_dict['tag'] = 'rc'
        >>> attr_dict
        {'openfed': 'openfed', 'version': 1.0, 'tag': 'rc'}
        >>> del attr_dict.tag
        >>> attr_dict
        {'openfed': 'openfed', 'version': 1.0}
        >>> del attr_dict['version']
        >>> attr_dict
        {'openfed': 'openfed'}
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class ArrayDict(dict):
    """Array Dictionary. Index items via int index.
    A Lock will add automatically to make sure that dictionary will not
    be modified in other thread. 

    .. Example::
        >>> array_dict = ArrayDict()
        >>> array_dict['a'] = [time_string(), 0]
        >>> array_dict['b'] = [time_string(), 1]
        >>> array_dict
        {'a': ['2021-07-19 23:21:13', 0], 'b': ['2021-07-19 23:21:24', 1]}
        >>> array_dict.default_value
        ['2021-07-19 23:21:13', 0]
        >>> array_dict.default_key
        'a'
        >>> len(array_dict)
        2
        >>> for i in array_dict:
        ...     print(i)
        ... 
        ('a', ['2021-07-19 23:21:13', 0])
        ('b', ['2021-07-19 23:21:24', 1])
        >>> array_dict[0]
        ('a', ['2021-07-19 23:21:13', 0])
        >>> array_dict[1]
        ('b', ['2021-07-19 23:21:24', 1])
        >>> array_dict[2]
        (None, None)
        >>> with array_dict:
        ...     array_dict['c'] = [time_string(), 3]
        ... 
        >>> array_dict
        {'a': ['2021-07-19 23:21:13', 0], 'b': ['2021-07-19 23:21:24', 1], 'c': ['2021-07-19 23:24:02', 3]}
    """

    def __init__(self, *args, **kwargs):
        # assign lock and index before init called
        self._lock = Lock()
        self._index = -1
        self._current_key = None
        self._current_value = None
        super().__init__(*args, **kwargs)

    @property
    def default_key(self) -> Any:
        return self[0][0]

    @property
    def default_value(self) -> Any:
        return self[0][1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if isinstance(index, int):
            # Rectify index
            index = len(self) - 1 if index > len(self) else index

            self._current_keys = list(self.keys())[
                index] if index < len(self) else None
            self._current_values = list(self.values())[
                index] if index < len(self) else None

            return self._current_keys, self._current_values
        else:
            return super().__getitem__(index)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Any, Any]:
        self._index = self._index + 1
        if self._index >= len(self):
            self._index = -1
            raise StopIteration
        else:
            return self[self._index]

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, exc_type, exc_value, trace):
        self._lock.release()


class TaskInfo(AttrDict):
    """A class to store different task inforation, such as instance number, 
    train and test accuracy.

    .. Example::
        >>> task_info = TaskInfo(part_id=1)
        >>> task_info
        {'part_id': 1}
        >>> task_info['accuracy'] = 0.98
        >>> task_info.accuracy
        0.98
        >>> task_info
        {'part_id': 1, 'accuracy': 0.98}
        >>> task_info.instance = 100
        >>> task_info['instance']
        100
        >>> task_info
        {'part_id': 1, 'accuracy': 0.98, 'instance': 100}
        >>> del task_info['instance']
        >>> task_info
        {'part_id': 1, 'accuracy': 0.98}
        >>> del task_info.accuracy
        >>> task_info
        {'part_id': 1}
    
    .. note::
        TaskInfo will have a default attribute of `train`.
        It will be used to control some aggregating process.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add a train flag to task info.
        if 'train' not in self:
            self.train = True

    def __str__(self):
        return openfed_class_fmt.format(
            class_name='TaskInfo',
            description=tablist(
                list(self.keys()),
                list(self.values()),
                items_per_row=6,
            )
        )

class Peeper(AttrDict):
    """Peeper to record all global vars.
    """
    def __str__(self):
        return openfed_class_fmt.format(
            class_name='Peeper',
            description=tablist(
                list(self.keys()),
                list(self.values()),
            ),
        )

peeper = Peeper()
peeper['api'] = None