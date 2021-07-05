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


from datetime import timedelta
from time import time
from typing import Union

import torch
from openfed.common.logging import logger
from torch.optim.lr_scheduler import _LRScheduler

from ..step import AtLast



class Aggregate(AtLast):
    checkpoint: Union[None, str]

    def __init__(self, lr_scheduler: _LRScheduler = None):
        super().__init__()
        self.lr_scheduler = lr_scheduler

    def aggregate(self, backend, *args, **kwargs):
        """Aggregate received models.
        """
        # Agg
        task_info_list = []
        for agg, optimizer in zip(backend.aggregator, backend.optimizer):
            # Zero grad first
            optimizer.zero_grad()

            # Aggregate
            agg.aggregate()

            # Unpack state from agg
            agg.unpack_state(optimizer)

            # Update models
            optimizer.step()

            # Clear buffers
            agg.clear_buffer()

        task_info_list = [reducer.reduce() for reducer in backend.reducer]
        [reducer.clear_buffer() for reducer in backend.reducer]

        backend.task_info_list = task_info_list

        for task_info in task_info_list:
            logger.success(f"Reduced:\n{task_info}")

        # update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Reset same flags
        backend.received_numbers = 0
        logger.success(
            f"-> @{backend.version} >> @{backend.version+1}.")
        backend.version += 1

        if self.checkpoint:
            path = f"{self.checkpoint}.{backend.version}"
            torch.save(backend.state_dict, path)
            logger.success(f"Save to {path}.")


class AggregatePeriod(Aggregate):
    tic: float

    def __init__(self, period: timedelta, checkpoint: str = None, lr_scheduler: _LRScheduler = None):
        """
        Args: 
            period: The period to agg received model.
            checkpoint: If specified, the new aggregated model will be saved as this checkpoint file.
        """
        super().__init__(lr_scheduler)
        self.period     = period
        self.tic        = time.time()
        self.checkpoint = checkpoint

    def step(self, backend, *args, **kwargs) -> None:
        toc = time.time()
        if timedelta(seconds=toc - self.tic) >= self.period:
            self.aggregate(backend, *args, **kwargs)
            # Update tic times.
            self.tic = time.time()
        else:
            pass


class AggregateCount(Aggregate):
    count: int

    def __init__(self, count: int, checkpoint: str = None, lr_scheduler: _LRScheduler = None):
        """
        Args:
            count: when the number of received models reach count, agg.
            checkpoint: if given, save the new aggregated model.
        """
        super().__init__(lr_scheduler)
        self.count      = count
        self.checkpoint = checkpoint

    def step(self, backend, *args, **kwargs) -> None:
        if backend.received_numbers >= self.count:
            self.aggregate(backend, *args, **kwargs)
        else:
            pass