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
import random
from ..step import MultiStep
from typing import Union, Any, Tuple
import time

class Dispatch(MultiStep):
    pending_queue: List[int]

    # part_id -> [nick name, time]
    running_queue : Dict[int, Tuple[str, float]]
    finished_queue: Dict[int, Tuple[str, float]]

    def __init__(self,
                 samples        : int,
                 parts_list     : Union[int, List[Any]],
                 test_samples   : int = None,
                 test_parts_list: Union[int, List[Any]] = None
                 ):
        """
        Args:
            samples: the total number of parts used in a train round.
            parts_list: a list contains all part ids.
            test_samples: the total number of parts used in a test round.
            test_parts_list: a list contains all part ids of test.
        """
        self.samples    = samples
        self.parts_list = list(range(parts_list)) if isinstance(
            parts_list, int) else parts_list
        self.test_samples    = test_samples
        self.test_parts_list = list(range(test_parts_list)) if isinstance(
            test_parts_list, int) else test_parts_list

        # Count the finished parts
        # If finished all parts in this round, reset inner part buffer.
        self._after_download()
        # Dispatch a part to be finished.
        self._before_upload()

        # Support auto register method.
        super().__init__()

        self.train = False

        # Initialize queue
        self.reset()

    def reset(self):
        self.train = not self.train if self.test_samples is not None else True

        if self.train:
            self.pending_queue  = random.sample(self.parts_list, self.samples)
            self.running_queue  = dict()
            self.finished_queue = dict()
        else:
            if self.test_samples is not None and self.test_parts_list is not None: 
                self.pending_queue = random.sample(
                    self.test_parts_list, self.test_samples)
                self.running_queue  = dict()
                self.finished_queue = dict()

    def after_download(self, backend, flag: bool):
        if flag:
            task_info = backend.delivery_task_info
            part_id   = task_info.part_id

            # pop from running queue
            nick_name, tic = self.running_queue.pop(part_id)
            toc = time.time()

            # add to finished queue
            self.finished_queue[part_id] = (nick_name, toc-tic)

            logger.debug(f"Received: from {backend.nick_name}, duration: {toc-tic:.2f} seconds.\n{task_info}")

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
            self.running_queue[part_id] = (backend.nick_name, time.time())

            # generate task_info
            task_info         = TaskInfo()
            task_info.part_id = part_id # type: ignore
            task_info.version = backend.version # type: ignore
            # opside with self.train
            task_info.train = self.train # type: ignore

            # set task_info
            backend.delivery_task_info = task_info

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
                head = ["Pending", "Running", "Finished"],
                data = [len(self.pending_queue), len(
                    self.running_queue), len(self.finished_queue)]
            )
        )
