from copy import copy
from threading import Lock
from typing import Any, List, Tuple

from openfed.utils import convert_to_list, openfed_class_fmt, tablist


class Buffer(object):
    """Used for Optimizer/Aggregator format class to clear buffer
    """
    param_groups: Any
    state: Any

    def clear_buffer(self, keep_keys: List[str] = None):
        """Clear state buffers.
        Args:
            keep_keys: if not specified, we will directly remove all buffers.
                Otherwise, the key in keep_keys will be kept.
        """
        keep_keys = convert_to_list(keep_keys)

        for group in self.param_groups:
            if 'keep_keys' in group:
                if keep_keys is None:
                    keys = group['keep_keys']
                elif group['keep_keys'] is None:
                    keys = keep_keys
                else:
                    keys = keep_keys + group['keep_keys']
            else:
                keys = keep_keys

            for p in group["params"]:
                if p in self.state[p]:
                    if keys is None:
                        del self.state[p]
                    else:
                        for k in self.state[p].keys():
                            if k not in keys:
                                del self.state[p][k]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class ArrayDict(dict):
    """
    ::Example
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
    """
    ::Example
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
    """

    def __str__(self):
        return openfed_class_fmt.format(
            class_name='TaskInfo',
            description=tablist(
                list(self.keys()),
                list(self.values()),
                force_in_one_row=True,
            )
        )

class Peeper(AttrDict):
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