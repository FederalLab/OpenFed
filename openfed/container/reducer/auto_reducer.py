from collections import defaultdict
from typing import List, Union

from openfed.common import TaskInfo
from openfed.utils import convert_to_list

from .reducer import Reducer


class AutoReducer(Reducer):
    """Auto reducer based on specified keys.
    """

    def __init__(self, reduce_keys: Union[str, List[str]] = None, weight_key: str = None):
        """
        Args:
            reduce_keys: if not specified, we will try to apply auto reduce on all int and float numbers.
            weight_key: if specified, we will apply a weighed reduce operation accross all values.
                weight_keys must be in the returned task_info_dict.
        """
        super().__init__()
        self.reduce_keys = convert_to_list(reduce_keys)
        self.weight_key  = weight_key

    def reduce(self) -> TaskInfo:
        task_info_list = self.task_info_buffer
        rdict          = defaultdict(lambda: 0.0)
        if self.weight_key is not None:
            task_info = task_info_list[0].info_dict
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
        return TaskInfo().load_dict(rdict)
