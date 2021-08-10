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


import time
from datetime import timedelta
from typing import Dict, List, Union

import torch
from openfed.common import logger, TaskInfo
from tqdm import trange

from .step import Step


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
        self.last_received_numbers = 0

        self.process_bar = self._process_bar(
            self.count[self.idx], description=self.count_name[self.idx])
        next(self.process_bar)

        self.max_loop_times = max_loop_times
        self.max_version = max_version

    def before_upload(self, leader, *args, **kwargs) -> bool:
        """Rewrite the before upload method. 
        In dispatch mode, we will response to client requests by task schedules.
        And the version information will be ignored. This is quiet different with Aggregate.
        """
        # version is not used in dispatch mode
        if leader.download_version <= leader.version:
            # generate task_info
            task_info = TaskInfo(
                version=leader.version,
                mode="federated learning",
            )
            # set task_info
            leader.delivery_task_info.update(task_info)

            return True
        else:
            # unknown case.
            return False

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
