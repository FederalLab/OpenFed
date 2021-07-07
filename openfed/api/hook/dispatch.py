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


from typing import Dict, List

import numpy as np
from openfed.common import TaskInfo, logger
from openfed.utils import openfed_class_fmt, tablist
from typing_extensions import overload

from ..step import MultiStep


class Dispatch(MultiStep):
    pending_queue: List[int]

    # part_id -> nick name
    running_queue: Dict[int, str]
    finished_queue: Dict[int, str]

    @overload
    def __init__(self, parts_list: int, samples: int = None, total_test_parts: int = None):
        """
        Args:
            total_parts: the total number of parts splitted in a simulation federated work.
            samples: the number of parts activated during simulation.
            total_test_parts: the number of test dataset parts.
        """

    @overload
    def __init__(self, total_parts: int, sample_ratio: float = None, total_test_parts: int = None):
        """
        Args:
            total_parts: the total number of parts splitted in a simulation federated work.
            sample_ratio: the ratio to be activated during simulation.
            total_test_parts: the number of test dataset parts.
        """

    def __init__(self, *args, **kwargs):
        total_parts = args[0]
        samples = kwargs.get('samples', None)
        sample_ratio = kwargs.get('sample_ratio', None)
        total_test_parts = kwargs.get('total_test_parts', None)

        # Count the finished parts
        # If finished all parts in this round, reset inner part buffer.
        self._after_download()
        # Dispatch a part to be finished.
        self._before_upload()

        # Support auto register method.
        super().__init__()

        assert not (
            samples is None and sample_ratio is None), "one of samples or sample_ratio must be specified."

        self.total_parts = total_parts
        self.samples = int(samples) if samples else int(
            total_parts * sample_ratio)
        self.total_test_parts = total_test_parts

        self.train = True

        # Initialize queue
        self.reset()

    def _permutation(self):
        return [int(x) for x in np.random.permutation(self.total_parts)[
            :self.samples]]

    def reset(self):
        if self.train:
            self.pending_queue = self._permutation()
            self.running_queue = dict()
            self.finished_queue = dict()
        else:
            self.pending_queue = list(range(self.total_test_parts))
            self.running_queue = dict()
            self.finished_queue = dict()
        # reverse
        self.train = not self.train

    def after_download(self, backend, flag: bool):
        if flag:
            task_info = backend.reign_task_info
            part_id = task_info.part_id

            logger.debug(f"Download a model from {backend.nick_name}.")

            # pop from running queue
            self.running_queue.pop(part_id)

            # add to finished queue
            self.finished_queue[part_id] = backend.nick_name

            # All finished
            if len(self.running_queue) == 0 and len(self.pending_queue) == 0:
                # Reset
                self.reset()
                logger.info(f"Start a new round.")
            else:
                logger.debug(self)

    def before_upload(self, backend, *args, **kwargs) -> bool:
        # version is not used in dispatch mode
        if len(self.pending_queue) > 0:
            # assign a new part id
            part_id = self.pending_queue.pop(-1)
            self.running_queue[part_id] = backend.nick_name

            # generate task_info
            task_info = TaskInfo()
            task_info.part_id = part_id
            task_info.version = backend.version
            # opside with self.train
            task_info.train = not self.train

            # set task_info
            backend.reign_task_info = task_info

            return True

        elif len(self.running_queue) > 0:
            # waiting other client to be finished.
            logger.debug(
                f"Waiting following client to submit there tasks: {list(self.running_queue.values())}.")
            return False
        else:
            # unknown case.
            return False

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Dispatch",
            description=tablist(
                head=["Pending", "Running", "Finished"],
                data=[len(self.pending_queue), len(
                    self.running_queue), len(self.finished_queue)]
            )
        )
