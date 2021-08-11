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


import json
from collections import defaultdict
from typing import List

from openfed.common import TaskInfo
from openfed.utils import convert_to_list

from .reduce_op import ReduceOp


class AutoReduceOp(ReduceOp):
    """Auto reducer based on specified keys.
    """

    def __init__(self,
                 reduce_keys: List[str] = ['accuracy'],
                 weight_key: str = None,
                 ignore_keys: List[str] = None,
                 log_file: str = None):
        """
        Args:
            weight_key: If specified, we will apply a weighed reduce operation accross all values.
                ``weight_key`` must be in the returned task_info_dict.
            reduce_keys: if not specified, auto reduce will be applied on all `int` and `float` numbers.

        .. note:: 
            The extra key value in task_info_dict will keep the same as first task info.
        """
        super().__init__()
        assert isinstance(weight_key, str), "weight_key must be a string."
        self.weight_key = weight_key
        self.reduce_keys = convert_to_list(reduce_keys) or []
        self.ignore_keys = convert_to_list(ignore_keys) or []

        assert self.reduce_keys, "Attempt to reduce empty list of keys."

        self.log_file = log_file
        # Clear file
        if self.log_file is not None:
            with open(self.log_file, 'w') as f:
                f.write('')

    def reduce(self) -> TaskInfo:
        """Reduce the task_info_buffer and then clear it.
        """
        task_info_list = self.task_info_buffer
        rdict = defaultdict(lambda: 0.0)
        task_info = task_info_list[0]
        if self.weight_key is not None:
            assert self.weight_key in task_info, "weight key is not contained in task info."
            demo = sum([ti[self.weight_key]
                       for ti in task_info_list])
            rdict[self.weight_key] = demo
            weight = [ti[self.weight_key] /
                      demo for ti in task_info_list]
        else:
            weight = [1.0/len(task_info_list)
                      for _ in range(len(task_info_list))]

        for w, ti in zip(weight, task_info_list):
            for k, v in ti.items():
                if k == self.weight_key:
                    # Skip weight key
                    continue
                elif k in self.reduce_keys:
                    rdict[k] += v * w
                elif k not in rdict:
                    # Keep unexpected value.
                    # They may be `version` information.
                    rdict[k] = v

        # Clear buffers
        self.task_info_buffer.clear()

        for k in self.ignore_keys:
            del rdict[k]

        r_task_info = TaskInfo(**rdict)

        # Log reduced task info to a json file.
        # You can plot the curve from this json file.
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                json.dump(r_task_info, f)

        return r_task_info
