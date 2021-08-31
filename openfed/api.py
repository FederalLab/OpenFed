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
from copy import copy
from threading import Lock
from typing import Any, Dict, List, Union

from torch import Tensor

from openfed.common import Attach, DeviceOffline, TaskInfo, logger, peeper
from openfed.core import FederatedGroupProperties, Pipe, init_federated_group
from openfed.hooks.collector import Collector
from openfed.hooks.cypher import Cypher
from openfed.hooks.step import (Step, after_destroy, after_download,
                                after_upload, at_failed, at_first,
                                at_invalid_state, at_last, at_new_episode,
                                at_zombie, before_destroy, before_download,
                                before_upload)
from openfed.optim import Aggregator, FedOptim
from openfed.utils import convert_to_list, openfed_class_fmt

peeper.api_lock = Lock()  # type: ignore


def federated_group(func):
    def _federated_group(self, *args, **kwargs):
        def safe_call(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DeviceOffline as e:
                return False

        if self.pipe.distributed_properties.lock.locked():
            return safe_call(self, *args, **kwargs)
        else:
            # If pipe lock is unlocked, we need to lock it.
            with self.pipe.distributed_properties:
                return safe_call(self, *args, **kwargs)

    return _federated_group


class API(Attach):
    """Provide a unified api for leader and role.
    """

    # Communication related
    pipe: Pipe
    pipes: List[Pipe]
    current_step: str

    def __init__(self,
                 state_dict: Dict[str, Tensor],
                 fed_optim: FedOptim,
                 aggregator: Aggregator = None):
        """Whether act as a role.
        Frontend is always in sync mode, which will ease the coding burden.
        Backend will be set as async mode by default.
        """
        super().__init__()
        # how many times for leader waiting for connections.

        # Set default value
        self.version: int = 0

        self._hooks_cypher: List[Cypher] = []
        self._hooks_collector: List[Collector] = []

        self.stopped: bool = False
        self.received_numbers: int = 0
        self.last_aggregate_time: float = time.time()
        self.pipe_task_info: TaskInfo = TaskInfo()
        self.task_info_list: List[TaskInfo] = []

        # Data handle
        self.state_dict: Dict[str, Tensor] = state_dict
        self.aggregator: List[Aggregator] = convert_to_list(aggregator)
        self.fed_optim: List[FedOptim] = convert_to_list(fed_optim)

        self.pipes = []

    def register_everything(self, hook: Union[Step, Collector, Cypher]):
        """register hook to the corresponding class based on different hook types.
        """
        hook.api = self

        if isinstance(hook, Step):
            """Register the step function to step possition call."""
            self.register_hook(hook)
        elif isinstance(hook, Collector):
            self._hooks_collector.append(hook)
        elif isinstance(hook, Cypher):
            self._hooks_cypher.append(hook)
        else:
            raise NotImplementedError(
                f'Hook type is not supported: {type(hook)}.')

    def _add_hook_to_pipe(self):
        # register a clone of informer hook.
        # informer hook may contain some inner variable, which is not allowed
        # to share with each other.
        [
            self.pipe.register_collector(copy(hook))
            for hook in self._hooks_collector
            if hook.bounding_name not in self.pipe._hook_dict
        ]
        # register the hook directly.
        # deliver hook is not allowed to have inner parameters.
        # it can be used among all pipe.
        [
            self.pipe.register_cypher(hook) for hook in self._hooks_cypher
            if hook not in self.pipe._hook_list
        ]

    def build_connection(self,
                         federated_group_properties: FederatedGroupProperties):
        self.pipes += init_federated_group(federated_group_properties)
        for pipe in self.pipes:
            self.pipe = pipe
            self._add_hook_to_pipe()

    def update_version(self, version: int = None):
        """Update inner model version.
        """
        self.version = version if version is not None else self.version + 1

    @federated_group
    def transfer(self, to: bool = False, task_info: TaskInfo = None) -> bool:
        r"""Transfer inner state to other ends.
        Args:
            to: if true, upload data to the other end.
                if false, download data from the other end.
            task_info: task info want to upload or download.
                The new task info will be updated directly to this parameters.
        """
        # Collect system information from other end.
        # Most of the collect function will only be called once.
        self.pipe.collect()

        # Scatter system information to other end.
        # Most of the scatter function will only be called once.
        self.pipe.scatter()

        # Reset state
        self.pipe.reset_state_dict(self.state_dict)

        # Push data to the other end.
        if to:
            # Assign task info to other end
            self.pipe_task_info = task_info or self.pipe_task_info
            self.pipe.set_task_info(self.pipe_task_info)

            # Pack related inner state of fed_optim.
            [self.pack_state(fed_optim) for fed_optim in self.fed_optim]
            if self.leader:
                [self.pack_state(agg) for agg in self.aggregator]

            # Upload data automatically.
            flag = self.pipe.upload(self.version)

        # Pull data from the other end.
        else:
            # Download data automatically.
            flag = self.pipe.download(self.version)

        if flag and not to:
            # update task info
            self.pipe_task_info = self.pipe.task_info
            if self.follower:
                # Reset current version as downloaded version.
                self.version = self.pipe_task_info.version  # type: ignore
                # As for follower, we will unpack the inner state from
                # received tensor automatically.
                [self.unpack_state(fed_optim) for fed_optim in self.fed_optim]
            elif self.leader:
                if self.pipe_task_info.mode == 'train':  # type: ignore
                    if self.upload_version <= self.version:
                        # In federated learning, some device may upload the outdated model.
                        # This is not desired, we should skip this invalid model.
                        logger.warning(
                            f"Received version of model is outdate."
                            f"(Expected: > @{self.version}, Received: @{self.upload_version})."
                        )
                    else:
                        # As for leader, we will increase the received numbers
                        # and catch received tensor to aggregator.
                        self.received_numbers += 1
                        packages = self.tensor_indexed_packages
                        [
                            aggregator.step(packages, self.pipe_task_info)
                            for aggregator in self.aggregator
                        ]
                else:
                    # If under test mode it is not necessary to do the aggregation operation.
                    # As for leader, we will increase the received numbers
                    # and catch received tensor to aggregator.
                    self.received_numbers += 1
                    [
                        aggregator.step({}, self.pipe_task_info)
                        for aggregator in self.aggregator
                    ]

            if task_info is not None:
                # Update the task info if necessary.
                task_info.update(self.pipe_task_info)

        return flag

    def step(self, *args, **kwargs):
        if self.follower:
            # upload and download
            download = kwargs.pop('download', True)
            upload = kwargs.pop('upload', True)
            task_info = kwargs.pop('task_info', None)

            if upload:
                self.transfer(to=True, task_info=task_info)

            if download:
                self.transfer(to=False, task_info=task_info)
        else:

            self.stopped = False

            def step(step_name: str, *args, **kwargs) -> Union[None, bool]:
                """
                    You can chain the same type hook together.
                    Hook will return a bool value or None.
                    If bool is returned, we will use `and` to reduce them.
                    If None is returned, we will return `None` directly.
                    You should directly store other variables in self object.
                """
                self.current_step = step_name
                output = [
                    hook(self, step_name, *args, **kwargs)
                    for hook in self.hook_list
                ]

                # reduce output
                if False in output:
                    return False
                elif True in output:
                    return True
                else:
                    return None

            while not self.stopped and len(self.pipes) > 0:
                step(at_new_episode)
                cnt = 0
                for i, pipe in enumerate(self.pipes):
                    if self.stopped:
                        break
                    # assign pipe to self first.
                    self.pipe = pipe

                    cnt += 1
                    step(at_first)
                    if pipe.is_offline:
                        step(before_destroy)
                        del self.pipes[i]
                        step(after_destroy, True)
                    elif pipe.is_zombie:
                        step(at_zombie)
                    elif pipe.is_pushing:
                        # Follower want to push data to leader, we need to download.
                        if step(before_download):
                            step(after_download, self.transfer(to=False))
                        else:
                            step(at_failed)
                    elif pipe.is_pulling:
                        # Follower want to pull data from leader, we need to upload
                        if step(before_upload):
                            step(after_upload, self.transfer(to=True))
                        else:
                            step(at_failed)
                    else:
                        step(at_invalid_state)
                    # update regularly.
                    step(at_last)

                # Sleeping a while to wait the state being stable.
                time.sleep(0.1)

    def finish(self, auto_exit: bool = False):
        """Kill all pipe in this world as soon as possible.
        """
        self.manual_stop()

        self.pipes.clear()

        if auto_exit and self.leader:
            exit(15)

    def manual_stop(self):
        self.stopped = True

    @property
    def tensor_indexed_packages(self) -> Any:
        return self.pipe.tensor_indexed_packages

    @property
    def pack_state(self):
        return self.pipe.pack_state

    @property
    def nick_name(self):
        return self.pipe.nick_name

    @property
    def unpack_state(self):
        return self.pipe.unpack_state

    @property
    def upload_version(self):
        return self.pipe.upload_version

    @property
    def download_version(self):
        return self.pipe.download_version

    @property
    def leader(self):
        return self.pipe.leader

    @property
    def follower(self):
        return self.pipe.follower

    @property
    def role(self):
        return self.pipe.role

    def __del__(self):
        self.finish(auto_exit=False)

    def __enter__(self):
        peeper.api_lock.acquire()  # type: ignore
        peeper.api = self  # type: ignore

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore state
        peeper.api = None  # type: ignore
        peeper.api_lock.release()  # type: ignore
