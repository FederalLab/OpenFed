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


import random
import time
from typing import Dict, List, Tuple

from openfed.common import TaskInfo, logger
from openfed.common.logging import logger
from openfed.utils import openfed_class_fmt, tablist
from datetime import timedelta

from .aggregate import Aggregate


class Dispatch(Aggregate):
    pending_queue: List[int]

    # part_id -> [nick name, time]
    running_queue: Dict[int, Tuple[str, float]]
    finished_queue: Dict[int, Tuple[str, float]]

    timeout: float = -1

    def __init__(self,
                 activated_parts: Dict[str, int],
                 parts_list: Dict[str, List],
                 period: timedelta = timedelta(hours=24),
                 checkpoint: str = None,
                 max_loop_times: int = -1,
                 max_version: int = -1,
                 **kwargs
                 ):
        """
        Args:
            *args: parameters for aggregate.
            parts_list: a list contains all part ids.
            test_samples: the total number of parts used in a test round.
            test_parts_list: a list contains all part ids of test.
            **kwargs: parameters for aggregate.
        """
        for k in activated_parts:
            if k not in parts_list:
                raise KeyError(f"parts_list must contain {k}")

        super().__init__(
            activated_parts, period, checkpoint, max_loop_times, max_version, **kwargs)

        self.parts_list = parts_list

        # Initialize queue
        self.reset()

    def reset(self):
        key = self.stage_name

        self.pending_queue = random.sample(
            self.parts_list[key], self.activated_parts[key])
        self.finished_queue = dict()
        self.running_queue = dict()

    def after_download(self, leader, flag: bool):
        if flag:
            task_info = leader.delivery_task_info
            part_id = task_info.part_id

            # pop from running queue
            if part_id not in self.running_queue:
                logger.error(f"Invalid part id: {part_id}")
                return
            nick_name, tic = self.running_queue.pop(part_id)
            toc = time.time()

            # add to finished queue
            self.finished_queue[part_id] = (nick_name, toc-tic)

            logger.debug(
                f"Received: from {leader.nick_name}, duration: {toc-tic:.2f} seconds.\n{task_info}")

            # All finished
            if len(self.running_queue) == 0 and len(self.pending_queue) == 0:
                logger.info(f"Start a new round.")
            else:
                logger.debug(self)

    def before_upload(self, leader, *args, **kwargs) -> bool:
        """Rewrite the before upload method. 
        In dispatch mode, we will response to client requests by task schedules.
        And the version information will be ignored. This is quiet different with Aggregate.
        """
        # version is not used in dispatch mode
        if len(self.pending_queue) > 0:
            # assign a new part id
            part_id = self.pending_queue.pop(-1)
            self.running_queue[part_id] = (leader.nick_name, time.time())

            # generate task_info
            task_info = TaskInfo(
                part_id=part_id,
                version=leader.version,
                mode=self.stage_name,
            )
            # set task_info
            leader.delivery_task_info.update(task_info)

            return True

        elif len(self.running_queue) > 0:
            timeout_parts = []  # [part_id, nick_name, duration]
            for part_id, (nick_name, assign_time) in self.running_queue.items():
                if self.timeout > 0 and time.time() - assign_time > self.timeout:
                    # resign this task
                    self.pending_queue.append(part_id)
                    timeout_parts.append(
                        (part_id, nick_name, time.time() - assign_time))
            for part, _, _ in timeout_parts:
                del self.running_queue[part]

            if len(timeout_parts) > 0:
                logger.error(f"Timeout task: {timeout_parts}")

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
