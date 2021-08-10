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
from threading import Lock, Thread
from typing import Any, Dict, List, Union

from torch import Tensor

from openfed.common import (Address, Attach, DeviceOffline, TaskInfo,
                            default_tcp_address, logger, peeper)
from openfed.core import Pipe, Destroy, Maintainer, World, openfed_lock
from openfed.hooks.collector import Collector
from openfed.hooks.cypher import Cypher
from openfed.hooks.step import (Step, after_destroy, after_download,
                                after_upload, at_failed, at_first,
                                at_invalid_state, at_last, at_new_episode,
                                at_zombie, before_destroy, before_download,
                                before_upload)
from openfed.optim import FedOptim, Aggregator
from openfed.utils import (convert_to_list, keyboard_interrupt_handle,
                           openfed_class_fmt)

peeper.api_lock = Lock() # type: ignore


def device_offline_care(func):
    """Return False instead of raise an exception when device is offline.

    .. warn::
        This decorator is unable to catch the exception raised by `assert`.
    """
    def _device_offline_care(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DeviceOffline as e:
            logger.error(e)
            return False
    return _device_offline_care


class API(Thread, Attach):
    """Provide a unified api for leader and role.
    """

    # Communication related
    maintainer: Maintainer
    pipe: Pipe
    current_step: str

    def __init__(self,
                 world: World,
                 state_dict: Dict[str, Tensor],
                 fed_optim: FedOptim,
                 aggregator: Aggregator = None,
                 ):
        """Whether act as a role.
        Frontend is always in sync mode, which will ease the coding burden.
        Backend will be set as async mode by default.
        """
        super().__init__(daemon=True)
        self.world = world

        # how many times for leader waiting for connections.

        keyboard_interrupt_handle()

        # Set default value
        self.version: int = 0

        self._hooks_cypher: List[Cypher] = []
        self._hooks_collector: List[Collector] = []

        self.stopped: bool = False
        self.received_numbers: int = 0
        self.last_aggregate_time: float = time.time()
        self.delivery_task_info: TaskInfo = TaskInfo()
        self.task_info_list: List[TaskInfo] = []

        # Data handle
        self.state_dict: Dict[str, Tensor] = state_dict
        self.aggregator: List[Aggregator] = convert_to_list(aggregator)
        self.fed_optim: List[FedOptim] = convert_to_list(fed_optim)

    def register_everything(self, hook: Union[Step, Collector, Cypher]):
        """register hook to the corresponding class based on different hook types.
        """
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

    def _add_hook_to_delivery(self):
        # register a clone of informer hook.
        # informer hook may contain some inner variable, which is not allowed
        # to share with each other.
        [self.pipe.register_collector(copy(
            hook)) for hook in self._hooks_collector if hook.bounding_name not in self.pipe._hook_dict]
        # register the hook directly.
        # deliver hook is not allowed to have inner parameters.
        # it can be used among all pipe.
        [self.pipe.register_cypher(
            hook) for hook in self._hooks_cypher if hook not in self.pipe._hook_list]

    def build_connection(self, address: Union[Address, List[Address]] = None, address_file: str = None):
        world = self.world
        if address is None and address_file is None:
            address = default_tcp_address

        if self.leader and self.world.dal:
            # NOTE: hold openfed_lock before create a dynamic address loading maintainer.
            # otherwise, it may interrupt the process and cause error before you go into loop()
            openfed_lock.acquire()
        self.maintainer = Maintainer(
            world, address=address, address_file=address_file)  # type: ignore

        if self.follower:
            self.pipe = Pipe.default_delivery()
            self._add_hook_to_delivery()

    def update_version(self, version: int = None):
        """Update inner model version.
        """
        self.version = version if version is not None else self.version + 1

    @device_offline_care
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
            self.delivery_task_info = task_info or self.delivery_task_info
            self.pipe.set_task_info(self.delivery_task_info)

            # Pack related inner state of fed_optim.
            [self.pack_state(fed_optim) for fed_optim in self.fed_optim]

            # Upload data automatically.
            flag = self.pipe.upload(self.version)

        # Pull data from the other end.
        else:
            # Download data automatically.
            flag = self.pipe.download(self.version)

        if flag and not to:
            # update task info
            self.delivery_task_info = self.pipe.task_info
            if self.follower:
                # Reset current version as downloaded version.
                self.version = self.delivery_task_info.version # type: ignore
                # As for follower, we will unpack the inner state from 
                # received tensor automatically.
                [self.unpack_state(fed_optim) for fed_optim in self.fed_optim]
            elif self.leader:
                if self.delivery_task_info.mode == 'train': # type: ignore
                    if self.upload_version <= self.version:
                        # In federated learning, some device may upload the outdated model.
                        # This is not desired, we should skip this invalid model.
                        logger.warning(
                            f"Received version of model is outdate."
                            f"(Expected: > @{self.version}, Received: @{self.upload_version}).")
                    else:
                        # As for leader, we will increase the received numbers 
                        # and catch received tensor to aggregator.
                        self.received_numbers += 1
                        packages = self.tensor_indexed_packages
                        [aggregator.step(packages, self.delivery_task_info)
                        for aggregator in self.aggregator]
                else:
                    # If under test mode it is not necessary to do the aggregation operation.
                    # As for leader, we will increase the received numbers 
                    # and catch received tensor to aggregator.
                    self.received_numbers += 1
                    [aggregator.step({}, self.delivery_task_info)
                        for aggregator in self.aggregator]

            if task_info is not None:
                # Update the task info if necessary.
                task_info.update(self.delivery_task_info)

        return flag

    def run(self):
        """
            Use self.run() to start this loop in the main thread.
            Use self.start() to start this loop in the thread.
        """
        # NOTE: release openfed_lock here.
        if self.world.dal:
            openfed_lock.release()

        if self.follower:
            return None

        def step(step_name: str, *args, **kwargs) -> Union[None, bool]:
            """
                You can chain the same type hook together.
                Hook will return a bool value or None.
                If bool is returned, we will use `and` to reduce them.
                If None is returned, we will return `None` directly.
                You should directly store other variables in self object.
            """
            self.current_step = step_name
            output = [hook(self, step_name, *args, **kwargs) for hook in self.hook_list]

            # reduce output
            if False in output:
                return False
            elif True in output:
                return True
            else:
                return None

        try_times = 0
        while not self.stopped and try_times < self.world.mtt:
            with self.maintainer.pending_queue:
                step(at_new_episode)
                cnt = 0
                for pipe in Pipe.delivery_generator():
                    if self.stopped:
                        break
                    # assign pipe to self first.
                    self.pipe = pipe

                    # register hook to pipe if necessary.
                    self._add_hook_to_delivery()

                    cnt += 1
                    step(at_first)
                    if pipe.is_offline:
                        [step(after_destroy, Destroy.destroy_delivery(pipe)) if step(
                            before_destroy) else step(at_failed)]
                    elif pipe.upload_hang_up:
                        step(after_upload,
                             pipe.deal_with_hang_up())
                    elif pipe.download_hang_up:
                        step(after_download,
                             pipe.deal_with_hang_up())
                    elif pipe.is_zombie:
                        step(at_zombie)
                    elif pipe.is_pushing:
                        # Follower want to push data to leader, we need to download.
                        [step(after_download, self.transfer(to=False)) if step(
                            before_download) else step(at_failed)]
                    elif pipe.is_pulling:
                        # Follower want to pull data from leader, we need to upload
                        [step(after_upload, self.transfer(to=True)) if step(
                            before_upload) else step(at_failed)]
                    else:
                        step(at_invalid_state)
                    # update regularly.
                    step(at_last)
                    # sleep a short time is very import!
                    # otherwise, some states may be rewrite.
                    time.sleep(0.1)
            if cnt == 0:
                time.sleep(5.0)
                try_times  += 1
            else:
                time.sleep(0.1)
                try_times = 0

    def finish(self, auto_exit: bool = False):
        """Kill all pipe in this world as soon as possible.
        """
        if self.maintainer:
            # Stop maintainer first
            self.maintainer.manual_stop()
            self.maintainer.join()

            # Delete the pipe in world
            world = self.maintainer.world
            for pipe in list(world._delivery_dict.keys()):
                Destroy.destroy_delivery(pipe)

        if auto_exit and self.leader:
            exit(15)

    def manual_stop(self):
        self.stopped = True

    @property
    def role(self) -> str:
        return self.world.role

    @property
    def leader(self) -> bool:
        return self.world.leader

    @property
    def follower(self) -> bool:
        return self.world.follower

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

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="OpenFedAPI",
            description=f"ROLE: {self.role}"
        )

    def __enter__(self):
        peeper.api_lock.acquire() # type: ignore
        peeper.api = self # type: ignore

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore state
        peeper.api = None # type: ignore
        peeper.api_lock.release() # type: ignore
