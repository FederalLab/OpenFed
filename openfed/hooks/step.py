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
from datetime import timedelta
from typing import Dict, List, Tuple, Union

import torch
from openfed.common import TaskInfo, logger
from openfed.common.logging import logger
from openfed.utils import convert_to_list, openfed_class_fmt, tablist
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange

from .hooks import Hooks

after_destroy = 'AFTER_DESTROY'
after_download = 'AFTER_DOWNLOAD'
after_upload = 'AFTER_UPLOAD'

at_first = "AT_FIRST"
at_failed = 'AT_FAILED'
at_invalid_state = 'AT_INVALID_STATE'
at_last = 'AT_LAST'
at_new_episode = 'AT_NEW_EPISODE'
at_zombie = 'AT_ZOMBIE'

before_destroy = 'BEFORE_DESTROY'
before_download = 'BEFORE_DOWNLOAD'
before_upload = 'BEFORE_UPLOAD'


class Step(Hooks):
    """Step hook used for openfed_api.
    """

    def before_destroy(self, leader, *args, **kwargs) -> bool:
        return True

    def before_download(self, leader, *args, **kwargs) -> bool:
        return True

    def before_upload(self, leader, *args, **kwargs) -> bool:
        return True

    def after_destroy(self, leader, *args, **kwargs):
        ...

    def after_download(self, leader, *args, **kwargs):
        ...

    def after_upload(self, leader, *args, **kwargs):
        ...

    def at_failed(self, leader, *args, **kwargs):
        ...

    def at_invalid_state(self, leader, *args, **kwargs):
        ...

    def at_last(self, leader, *args, **kwargs):
        ...

    def at_first(self, leader, *args, **kwargs):
        ...

    def at_new_episode(self, leader, *args, **kwargs):
        ...

    def at_zombie(self, leader, *args, **kwargs):
        ...

    def __call__(self, leader, step_name, *args, **kwargs):
        func = getattr(self, step_name.lower())
        return func(leader, *args, **kwargs)


class Aggregate(Step):
    checkpoint: Union[None, str]
    tic: float
    count: List[int]
    max_version: int
    max_loop_times: int

    def __init__(self,
                 count: Dict[str, int] = dict(train=-1),
                 period: timedelta = timedelta(hours=24),
                 checkpoint: str = None,
                 lr_scheduler: Union[_LRScheduler, List[_LRScheduler]] = None,
                 max_loop_times: int = -1,
                 max_version: int = -1):
        """
        Args: 
            count: The circle times to do the aggregation operation.
            period: The period to agg received model.
            checkpoint: If specified, the new aggregated model will be saved as this checkpoint file.
            max_loop_times: if loop times exceed this number, we will stop the server.
            max_version: when inner version number achieves this number, we will stop server.
        """
        super().__init__()

        # Aggregate
        self.period = period
        self.count = list(count.values())
        self.count_name = list(count.keys())
        self.idx = 0

        self.tic = time.time()
        self.checkpoint = checkpoint
        self.lr_scheduler = convert_to_list(lr_scheduler)
        self.last_received_numbers = 0

        self.process_bar = self._process_bar(
            self.count[self.idx], description=self.count_name[self.idx])
        next(self.process_bar)

        self.max_loop_times = max_loop_times
        self.max_version = max_version

    def before_upload(self, leader, *args, **kwargs) -> bool:
        return leader.download_version <= leader.version

    def aggregate(self, leader, *args, **kwargs):
        """Aggregate received models.
        """
        task_info_list = []
        for aggregator, fed_optim in zip(leader.aggregator, leader.fed_optim):
            # Zero grad first
            fed_optim.zero_grad()

            task_info_list.append(aggregator.reduce())

            # Aggregate
            aggregator.aggregate(clear_buffer=False)

            # Unpack state from agg
            aggregator.unpack_state(fed_optim)

            # Update models
            fed_optim.step()

            # Clear buffers
            aggregator.clear_buffer()

            fed_optim.round()

        leader.task_info_list = task_info_list

        for task_info in task_info_list:
            logger.success(task_info)

        # update learning rate
        if self.lr_scheduler is not None:
            [lr_sch.step() for lr_sch in self.lr_scheduler if lr_sch is not None]

        # Clear the received_numbers flag
        leader.received_numbers = 0

        if self.checkpoint:
            path = f"{self.checkpoint}.{leader.version}"
            torch.save(leader.state_dict, path)
            logger.info(f"Save to {path}.")

    def _process_bar(self, count, description=''):
        process_bar = trange(count)
        process_bar.set_description(description)
        for _ in process_bar:
            yield

    def at_last(self, leader, *args, **kwargs) -> None:
        cnt = self.count[self.idx]

        if self.last_received_numbers != leader.received_numbers:
            # indicate that received a new model.
            # then, update the index.
            self.last_received_numbers = leader.received_numbers
            try:
                next(self.process_bar)
            except StopIteration:
                pass

        if cnt > 0 and leader.received_numbers >= cnt:
            self.aggregate(leader, *args, **kwargs)
            self.idx += 1
            if self.idx >= len(self.count):
                self.idx = 0
                leader.version += 1
            self.process_bar = self._process_bar(
                self.count[self.idx], description=self.count_name[self.idx])
        toc = time.time()
        if timedelta(seconds=toc - self.tic) >= self.period:
            self.aggregate(leader, *args, **kwargs)
            # Update tic times.
            self.tic = time.time()

        # Terminate or not
        if (self.max_version != -1 and leader.version >= self.max_version) or\
                (self.max_loop_times != -1 and leader.loop_times >= self.max_loop_times):
            leader.manual_stop()


class Dispatch(Aggregate):
    pending_queue: List[int]

    # part_id -> [nick name, time]
    running_queue: Dict[int, Tuple[str, float]]
    finished_queue: Dict[int, Tuple[str, float]]

    timeout: float = -1

    def __init__(self,
                 samples: Dict[str, int],
                 parts_list: Dict[str, List],
                 *args, **kwargs
                 ):
        """
        Args:
            *args: parameters for aggregate.
            samples: the total number of parts used in a train round.
            parts_list: a list contains all part ids.
            test_samples: the total number of parts used in a test round.
            test_parts_list: a list contains all part ids of test.
            **kwargs: parameters for aggregate.
        """
        super().__init__(*args, **kwargs)

        self.samples = samples
        self.parts_list = parts_list

        self.sample_key_idx = -1

        # Initialize queue
        self.reset()

    def reset(self):
        self.sample_key_idx += 1
        if self.sample_key_idx >= len(self.samples):
            self.sample_key_idx = 0
        key = list(self.samples.keys())[self.sample_key_idx]

        self.pending_queue = random.sample(
            self.parts_list[key], self.samples[key])
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
                # Reset
                self.reset()
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
                mode=list(self.samples.keys())[self.sample_key_idx],
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


steps = [
    Aggregate, Dispatch, Step
]
