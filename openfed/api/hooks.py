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
from enum import Enum, unique
from typing import Any, Dict, List, Tuple, Union

import torch
from openfed.common import Clone, TaskInfo, logger
from openfed.common.base import peeper
from openfed.common.logging import logger
from openfed.utils import (convert_to_list, openfed_class_fmt, process_bar,
                           tablist)
from torch.optim.lr_scheduler import _LRScheduler

from .utils import download_callback


@unique
class StepName(Enum):
    # After
    AFTER_DESTROY = 'AFTER_DESTROY'
    AFTER_DOWNLOAD = 'AFTER_DOWNLOAD'
    AFTER_UPLOAD = 'AFTER_UPLOAD'

    # At
    AT_FIRST = "AT_FIRST"
    AT_FAILED = 'AT_FAILED'
    AT_INVALID_STATE = 'AT_INVALID_STATE'
    AT_LAST = 'AT_LAST'
    AT_NEW_EPISODE = 'AT_NEW_EPISODE'
    AT_ZOMBIE = 'AT_ZOMBIE'

    # Before
    BEFORE_DESTROY = 'BEFORE_DESTROY'
    BEFORE_DOWNLOAD = 'BEFORE_DOWNLOAD'
    BEFORE_UPLOAD = 'BEFORE_UPLOAD'


after_destroy = StepName.AFTER_DESTROY.value
after_download = StepName.AFTER_DOWNLOAD.value
after_upload = StepName.AFTER_UPLOAD.value

at_first = StepName.AT_FIRST.value
at_failed = StepName.AT_FAILED.value
at_invalid_state = StepName.AT_INVALID_STATE.value
at_last = StepName.AT_LAST.value
at_new_episode = StepName.AT_NEW_EPISODE.value
at_zombie = StepName.AT_ZOMBIE.value

before_destroy = StepName.BEFORE_DESTROY.value
before_download = StepName.BEFORE_DOWNLOAD.value
before_upload = StepName.BEFORE_UPLOAD.value


class Step(Clone):
    step_name: str

    def __init__(self):
        # automatically register step hooks to backend
        if peeper.api is not None:
            peeper.api.register_everything(self)

    def __call__(self, backend, *args, **kwargs) -> Union[None, bool]:
        return self.step(backend, *args, **kwargs)

    def step(self, backend, *args, **kwargs) -> Union[None, bool]:
        raise NotImplementedError("Not implemented!")


class AfterDestroy(Step):
    step_name = after_destroy


class AfterDownload(Step):
    step_name = after_download


class AfterUpload(Step):
    step_name = after_upload


class AtFirst(Step):
    step_name = at_first


class AtFailed(Step):
    step_name = at_failed


class AtInvalidState(Step):
    step_name = at_invalid_state


class AtLast(Step):
    step_name = at_last


class AtNewEpisode(Step):
    step_name = at_new_episode


class AtZombie(Step):
    step_name = at_zombie


class BeforeDestroy(Step):
    step_name = before_destroy


class BeforeDownload(Step):
    step_name = before_download


class BeforeUpload(Step):
    step_name = before_upload


class MultiStep(Step):
    step_name: List[str] = []

    def _after_destroy(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(after_destroy)

    def after_destroy(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _after_download(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(after_download)

    def after_download(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _after_upload(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('after_upload')

    def after_upload(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _at_failed(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_failed)

    def at_failed(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _at_invalid_state(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_invalid_state)

    def at_invalid_state(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _at_last(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_last)

    def at_last(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _at_new_episode(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_new_episode)

    def at_new_episode(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _at_zombie(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_zombie)

    def at_zombie(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _before_destroy(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(before_destroy)

    def before_destroy(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _before_download(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(before_download)

    def before_download(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def _before_upload(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(before_upload)

    def before_upload(self, backend, *args, **kwargs):
        raise NotImplementedError("Not implemented!")

    def __call__(self, backend, *args, **kwargs):
        if backend.current_step in self.step_name:
            func = getattr(self, backend.current_step.lower())
            return func(backend, *args, **kwargs)
        else:
            return None


class Aggregate(AtLast):
    checkpoint: Union[None, str]
    tic: float
    count: List[int]

    def __init__(self,
                 count: Union[int, List[int]] = -1,
                 period: timedelta = timedelta(hours=24),
                 checkpoint: str = None,
                 lr_scheduler: Union[_LRScheduler, List[_LRScheduler]] = None):
        """
        Args: 
            count: The circle times to do the aggregation operation.
            period: The period to agg received model.
            checkpoint: If specified, the new aggregated model will be saved as this checkpoint file.
        """
        super().__init__()
        self.period = period
        self.count = convert_to_list(count)
        self.idx = 0

        self.tic = time.time()
        self.checkpoint = checkpoint
        self.lr_scheduler = convert_to_list(lr_scheduler)
        self.last_received_numbers = 0

    def aggregate(self, backend, *args, **kwargs):
        """Aggregate received models.
        """
        task_info_list = []
        for container, pipe in zip(backend.container, backend.pipe):
            # Zero grad first
            pipe.zero_grad()

            # Aggregate
            container.aggregate()

            task_info_list.append(container.reduce())

            # Unpack state from agg
            container.unpack_state(pipe)

            # Update models
            pipe.step()

            # Clear buffers
            container.clear_buffer()

            pipe.round()

        backend.task_info_list = task_info_list

        for task_info in task_info_list:
            logger.success(task_info)

        # update learning rate
        if self.lr_scheduler is not None:
            [lr_sch.step() for lr_sch in self.lr_scheduler if lr_sch is not None]

        # Reset same flags
        backend.received_numbers = 0

        if self.checkpoint:
            path = f"{self.checkpoint}.{backend.version}"
            torch.save(backend.state_dict, path)
            logger.info(f"Save to {path}.")

    def step(self, backend, *args, **kwargs) -> None:
        cnt = self.count[self.idx]
        if self.last_received_numbers != backend.received_numbers:
            self.last_received_numbers = backend.received_numbers
            logger.success('\n' +
                           process_bar(
                               self.last_received_numbers /
                               self.count[self.idx],
                               prefix=f"@{backend.version}",
                           ))

        if cnt > 0 and backend.received_numbers >= cnt:
            self.aggregate(backend, *args, **kwargs)
            self.idx += 1
            if self.idx >= len(self.count):
                self.idx = 0
                backend.version += 1
        toc = time.time()
        if timedelta(seconds=toc - self.tic) >= self.period:
            self.aggregate(backend, *args, **kwargs)
            # Update tic times.
            self.tic = time.time()


class Dispatch(MultiStep):
    pending_queue: List[int]

    # part_id -> [nick name, time]
    running_queue: Dict[int, Tuple[str, float]]
    finished_queue: Dict[int, Tuple[str, float]]

    def __init__(self,
                 samples: int,
                 parts_list: Union[int, List[Any]],
                 test_samples: int = None,
                 test_parts_list: Union[int, List[Any]] = None
                 ):
        """
        Args:
            samples: the total number of parts used in a train round.
            parts_list: a list contains all part ids.
            test_samples: the total number of parts used in a test round.
            test_parts_list: a list contains all part ids of test.
        """
        self.samples = samples
        self.parts_list = list(range(parts_list)) if isinstance(
            parts_list, int) else parts_list
        self.test_samples = test_samples
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
            self.pending_queue = random.sample(self.parts_list, self.samples)
            self.running_queue = dict()
            self.finished_queue = dict()
        else:
            if self.test_samples is not None and self.test_parts_list is not None:
                self.pending_queue = random.sample(
                    self.test_parts_list, self.test_samples)
                self.running_queue = dict()
                self.finished_queue = dict()

    def after_download(self, backend, flag: bool):
        if flag:
            task_info = backend.delivery_task_info
            part_id = task_info.part_id

            # pop from running queue
            nick_name, tic = self.running_queue.pop(part_id)
            toc = time.time()

            # add to finished queue
            self.finished_queue[part_id] = (nick_name, toc-tic)

            logger.debug(
                f"Received: from {backend.nick_name}, duration: {toc-tic:.2f} seconds.\n{task_info}")

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
            task_info = TaskInfo()
            task_info.part_id = part_id  # type: ignore
            task_info.version = backend.version  # type: ignore
            # opside with self.train
            task_info.train = self.train  # type: ignore

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
                head=["Pending", "Running", "Finished"],
                data=[len(self.pending_queue), len(
                    self.running_queue), len(self.finished_queue)]
            )
        )


class Download(AfterDownload):

    def step(self, backend, flag: bool) -> None:
        if flag:  # Download success
            # download is to check others upload version
            if backend.upload_version <= backend.version:
                logger.warning(
                    f"Excepted @{backend.version}, received @{backend.upload_version}, discard.")
                return

            download_callback(backend)

            logger.info(
                f"{backend.received_numbers} at v.{backend.version} from {backend.nick_name}.")
        else:
            logger.debug(
                f"Try to download {backend.received_numbers+1} failed.")


class Terminate(AtLast):
    max_version: int
    max_loop_times: int

    def __init__(self, max_loop_times: int = -1, max_version: int = -1):
        """
        Args:
            max_loop_times: if loop times exceed this number, we will stop the server.
            max_version: when inner version number achieves this number, we will stop server.
        """
        super().__init__()
        self.max_loop_times = max_loop_times
        self.max_version = max_version

    def step(self, backend, *args, **kwargs) -> None:
        if self.max_version != -1 and backend.version >= self.max_version:
            logger.info("Terminate! Max version achieves.")
            backend.manual_stop()
        if self.max_loop_times != -1 and backend.loop_times >= self.max_loop_times:
            logger.info("Terminate! Max loop times achieves.")
            backend.manual_stop()


class Upload(BeforeUpload):
    def step(self, backend, *args, **kwargs) -> bool:

        # Check version requirements
        # upload is to check other download version.
        if backend.download_version > backend.version:
            logger.warning(
                f"Version not aligned. (request @{backend.download_version}, but @{backend.version}).")
            # Version is not satisfied.
            return False

        return True
