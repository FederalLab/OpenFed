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
from turtle import back
from typing import List, Union

import torch
from openfed.common.logging import logger
from openfed.utils import convert_to_list
from torch.optim.lr_scheduler import _LRScheduler

from ..step import AtLast


class Aggregate(AtLast):
    checkpoint: Union[None, str]
    tic: float
    count: int

    def __init__(self, 
        count       : int                                     = -1,
        period      : timedelta                               = timedelta(hours=24),
        checkpoint  : str                                     = None,
        lr_scheduler: Union[_LRScheduler, List[_LRScheduler]] = None):
        """
        Args: 
            period: The period to agg received model.
            checkpoint: If specified, the new aggregated model will be saved as this checkpoint file.
        """
        super().__init__()
        self.period = period
        self.count = count

        self.tic = time.time()
        self.checkpoint = checkpoint
        self.lr_scheduler = convert_to_list(lr_scheduler)

    def step(self, backend, *args, **kwargs) -> None:
        if self.count > 0 and backend.received_numbers >= self.count:
            self.aggregate(backend, *args, **kwargs)

        toc = time.time()
        if timedelta(seconds=toc - self.tic) >= self.period:
            self.aggregate(backend, *args, **kwargs)
            # Update tic times.
            self.tic = time.time()

    def aggregate(self, backend, *args, **kwargs):
        """Aggregate received models.
        """
        if backend.pipe is None:
            pipe = [None for _ in range(len(backend.aggregator))]
        else:
            pipe = backend.pipe
    

        for aggregator, bk_optimizer, pipe in zip(backend.aggregator, backend.bk_optimizer, pipe):
            # Zero grad first
            bk_optimizer.zero_grad()

            # Aggregate
            aggregator.aggregate()

            # Unpack state from agg
            aggregator.unpack_state(bk_optimizer)

            # Pipe 
            if pipe is not None:
                pipe.step(frontend=False)

            # Update models
            bk_optimizer.step()

            # Clear buffers
            aggregator.clear_buffer()
            if pipe is not None:
                pipe.finish_round(frontend=False)

        if backend.reducer is not None:
            task_info_list = [reducer.reduce() for reducer in backend.reducer]
            [reducer.clear_buffer() for reducer in backend.reducer]
        else:
            task_info_list = []

        backend.task_info_list = task_info_list

        for task_info in task_info_list:
            logger.success(f"Reduced:\n{task_info}")

        # update learning rate
        if self.lr_scheduler is not None:
            [lr_sch.step() for lr_sch in self.lr_scheduler]

        # Reset same flags
        backend.received_numbers = 0
        logger.success(
            f"-> @{backend.version} >> @{backend.version+1}.")
        backend.version += 1

        if self.checkpoint:
            path = f"{self.checkpoint}.{backend.version}"
            torch.save(backend.state_dict, path)
            logger.success(f"Save to {path}.")
