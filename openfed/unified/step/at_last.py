from abc import abstractmethod
from datetime import timedelta
from time import time

import torch
from openfed.common.logging import logger
from torch.optim.lr_scheduler import _LRScheduler

from .base import Backend, Step


class AtLast(Step):
    step_name = 'at_last'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        ...


class Aggregate(AtLast):
    checkpoint: str

    def __init__(self, lr_scheduler: _LRScheduler = None):
        self.lr_scheduler = lr_scheduler

    def aggregate(self, backend: Backend, *args, **kwargs):
        """Aggregate received models.
        """
        # Aggregate
        task_info_list = []
        for aggregator, optimizer in zip(backend.aggregator, backend.optimizer):
            # Zero grad first
            optimizer.zero_grad()

            # Aggregate will calculate new grad
            task_info_list.append(aggregator.aggregate())

            # Unpack state from aggregator
            aggregator.unpack_state(optimizer)

            # Update models
            optimizer.step()

            # Clear buffers
            aggregator.clear_buffer()

        backend.task_info_list = task_info_list
        # update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Reset same flags
        backend.received_numbers = 0
        logger.info(f"Update version from @{backend.version} to @{backend.version+1}.")
        backend.version += 1

        if self.checkpoint:
            torch.save(backend.state_dict,
                       f"{self.checkpoint}.{backend.version}")


class AggregatePeriod(Aggregate):
    tic: float

    def __init__(self, period: timedelta, checkpoint: str = None, lr_scheduler: _LRScheduler = None):
        """
        Args: 
            period: The period to aggregate received model.
            checkpoint: If specified, the new aggregated model will be saved as this checkpoint file.
        """
        super().__init__(lr_scheduler)
        self.period = period
        self.tic = time.time()
        self.checkpoint = checkpoint

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        toc = time.time()
        if timedelta(seconds=toc - self.tic) >= self.period:
            logger.info("Aggregate operation triggered by period.")
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
            count: when the number of received models reach count, aggregate.
            checkpoint: if given, save the new aggregated model.
        """
        super().__init__(lr_scheduler)
        self.count = count
        self.checkpoint = checkpoint

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        if backend.received_numbers >= self.count:
            logger.info("Aggregate operation triggered by count.")
            self.aggregate(backend, *args, **kwargs)
        else:
            pass


class StopAtVersion(AtLast):
    max_version: int

    def __init__(self, max_version: int):
        """
        Args:
            max_version: when inner version number achieves this number, we will stop server.
        """
        super().__init__()
        self.max_version = max_version

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        if backend.version >= self.max_version:
            backend.manual_stop()
        else:
            pass


class StopAtLoopTimes(AtLast):
    max_loop_times: int

    def __init__(self, max_loop_times: int):
        """
        Args:
            max_loop_times: if loop times exceed this number, we will stop the server.
        """
        super().__init__()
        self.max_loop_times = max_loop_times

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        if backend.loop_times >= self.max_loop_times:
            backend.manual_stop()
        else:
            pass
