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


"""This file provides some useful functions fpr auto-reduce operation among received task infos.
"""

from collections import defaultdict
from typing import List, Union

from openfed.common import TaskInfo
from openfed.utils import convert_to_list


class Reducer(object):
    """The base class of different reducers.
    """
    task_info_buffer: List[TaskInfo]  # assigned from aggregator

    def reduce(self) -> TaskInfo:
        return TaskInfo()

class AutoReducer(Reducer):
    """Auto reducer based on specified keys.
    """

    def __init__(self,
                 weight_key     : str                   = None,
                 reduce_keys    : Union[str, List[str]] = None,
                 additional_keys: Union[str, List[str]] = None):
        """
        Args:
            reduce_keys: if not specified, we will try to apply auto reduce on all int and float numbers.
            weight_key: if specified, we will apply a weighed reduce operation accross all values.
                weight_keys must be in the returned task_info_dict.
            additional_keys: the keys which do not apply reduce operation, but contains some message to indicate the task.
                we will keep these keys in the reduced task info.
        """
        super().__init__()
        self.weight_key      = weight_key
        self.reduce_keys     = convert_to_list(reduce_keys)
        self.additional_keys = convert_to_list(additional_keys)

    def reduce(self) -> TaskInfo:
        task_info_list = self.task_info_buffer
        rdict          = defaultdict(lambda: 0.0)
        task_info      = task_info_list[0].info_dict
        if self.weight_key is not None:
            assert self.weight_key in task_info, "weight key is not contained in task info."

            demo = sum([ti.info_dict[self.weight_key]
                       for ti in task_info_list])
            rdict[self.weight_key] = demo
            weight = [ti.info_dict[self.weight_key] /
                      demo for ti in task_info_list]
        else:
            weight = [1.0/len(task_info_list)
                      for _ in range(len(task_info_list))]

        for w, ti in zip(weight, task_info_list):
            ti = ti.info_dict
            for k, v in ti.items():
                if k == self.weight_key:
                    # skip weight key
                    continue

                if self.reduce_keys is not None and k in self.reduce_keys:
                    rdict[k] += v * w
                elif self.reduce_keys is None and isinstance(v, float):
                    rdict[k] += v * w
                else:
                    pass

        r_task_info = TaskInfo()
        if self.additional_keys is not None:
            for k in self.additional_keys:
                r_task_info.load_dict({k: task_info[k]})

        # Clear buffers
        self.task_info_buffer = []

        return r_task_info.load_dict(rdict)


reducers = [Reducer, AutoReducer]
