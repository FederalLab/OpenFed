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
from threading import Lock
from typing import Any, Dict, List, Union

from torch import Tensor
from copy import copy
import openfed
from openfed.common import (Address, Hook, SafeThread, TaskInfo,
                            default_address, logger)
from openfed.common.base import DeviceOffline, peeper
from openfed.container import Container
from openfed.core import (ROLE, Delivery, Destroy, Maintainer, World, follower,
                          leader, openfed_lock)
from openfed.hooks.collector import Collector
from openfed.hooks.cypher import Cypher
from openfed.hooks.step import (Step, after_destroy, after_download,
                                after_upload, at_failed, at_first,
                                at_invalid_state, at_last, at_new_episode,
                                at_zombie, before_destroy, before_download,
                                before_upload)
from openfed.pipe import Pipe
from openfed.utils import (convert_to_list, keyboard_interrupt_handle,
                           openfed_class_fmt)

peeper.api_lock = Lock()


def device_offline_care(func):
    def _device_offline_care(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except DeviceOffline as e:
            logger.error(e)
            return False
    return _device_offline_care


class API(SafeThread, Hook):
    """Provide a unified api for leader and role.
    """

    # Communication related
    maintainer: Maintainer
    delivery: Delivery
    current_step: str

    def __init__(self,
                 role: str,
                 state_dict: Dict[str, Tensor],
                 pipe: Pipe,
                 container: Container = None,
                 dal: bool = True,
                 async_op: bool = True,
                 max_try_times: int = 5):
        """Whether act as a role.
        Frontend is always in sync mode, which will ease the coding burden.
        Backend will be set as async mode by default.
        """
        # Call SafeThread init function.
        SafeThread.__init__(self, daemon=True)
        Hook.__init__(self)

        # how many times for leader waiting for connections.
        self.max_try_times: int = max_try_times

        self.dal = dal
        self.role = role

        keyboard_interrupt_handle()

        # Enable async_op if this is leader.
        self.async_op: bool = async_op if self.role == follower else True

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
        self.container: List[Container] = convert_to_list(container)
        self.pipe: List[Pipe] = convert_to_list(pipe)

    def register_everything(self, hook: Any):
        """register hook to the corresponding class based on different hook types.
        """
        if isinstance(hook, Step):
            step = hook
            """Register the step function to step possition call."""
            for name in convert_to_list(step.step_name):
                cnt = 0
                for n in self.hook_dict.keys():
                    if n.startswith(name):
                        cnt = max(cnt, int(n.split(".")[-1]))
                else:
                    self.register_hook(f"{name}.{cnt+1}", step)
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
        [self.delivery.register_collector(copy(
            hook)) for hook in self._hooks_collector if hook.bounding_name not in self.delivery._hook_dict]
        # register the hook directly.
        # deliver hook is not allowed to have inner parameters.
        # it can be used among all delivery.
        [self.delivery.register_cypher(
            hook) for hook in self._hooks_cypher if hook not in self.delivery._hook_list]

    def build_connection(self, world: World = None, address: Union[Address, List[Address]] = None, address_file: str = None):
        world = world if world is not None else World(role=self.role)

        assert world.role == self.role

        if address is None and address_file is None:
            address = default_address

        if self.role == leader and openfed.DAL.is_dal:
            # NOTE: hold openfed_lock before create a dynamic address loading maintainer.
            # otherwise, it may interrupt the process and cause error before you go into loop()
            openfed_lock.acquire()
        self.maintainer = Maintainer(
            world, address=address, address_file=address_file)  # type: ignore

        if self.role == follower:
            self.delivery = Delivery.default_delivery()
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
        # 1. gather hook information
        self.delivery.collect()
        self.delivery.scatter()

        # 2. set state dict
        assert self.state_dict
        self.delivery.reset_state_dict(self.state_dict)

        if to:
            # 3. set task info
            self.delivery_task_info = self.delivery_task_info if task_info is None else task_info
            self.delivery.set_task_info(self.delivery_task_info)

            # 4. Pack state
            if self.pipe is not None:
                [self.pack_state(pipe) for pipe in self.pipe]

            # 5. transfer
            flag = self.delivery.upload(self.version)
        else:
            flag = self.delivery.download(self.version)

        def callback():
            if not to:
                self.delivery_task_info = self.delivery.task_info
                if self.role == follower:
                    [self.unpack_state(pipe) for pipe in self.pipe]
                elif self.role == leader:
                    # Increase the total number of received models
                    self.received_numbers += 1
                    packages = self.tensor_indexed_packages
                    [container.step(packages, self.delivery_task_info)
                     for container in self.container]

                if task_info is not None:
                    task_info.update(self.delivery_task_info)

        # 7. hand on
        if flag:
            callback()
            return True
        else:
            if self.role == leader:
                # return directly.
                return flag
            else:
                # wait for finished.
                if openfed.ASYNC_OP.is_async_op:
                    while not self.delivery.deal_with_hang_up():
                        if self.delivery.is_offline:
                            return False
                        time.sleep(0.1)
                    else:
                        callback()
                        return True
                else:
                    return False

    def safe_run(self):
        """
            Use self.run() to start this loop in the main thread.
            Use self.start() to start this loop in the thread.
        """
        # NOTE: release openfed_lock here.
        if openfed.DAL.is_dal:
            openfed_lock.release()

        if self.role is follower:
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
            output = [hook(self, *args, **kwargs) for name,
                      hook in self.hook_dict.items() if name.startswith(step_name)]

            # reduce output
            if None in output:
                return None
            elif False in output:
                return False
            else:
                return True

        try_times = 0
        while not self.stopped:
            with self.maintainer.pending_queue:
                step(at_new_episode)
                cnt = 0
                for delivery in Delivery.delivery_generator():
                    if self.stopped:
                        break
                    # assign delivery to self first.
                    self.delivery = delivery

                    # register hook to delivery if necessary.
                    self._add_hook_to_delivery()

                    cnt += 1
                    step(at_first)
                    if delivery.is_offline:
                        [step(after_destroy, Destroy.destroy_delivery(delivery)) if step(
                            before_destroy) else step(at_failed)]
                    elif delivery.upload_hang_up:
                        step(after_upload,
                             delivery.deal_with_hang_up())
                    elif delivery.download_hang_up:
                        step(after_download,
                             delivery.deal_with_hang_up())
                    elif delivery.is_zombie:
                        step(at_zombie)
                    elif delivery.is_pushing:
                        # Client want to push data to server, we need to download.
                        [step(after_download, self.transfer(to=False)) if step(
                            before_download) else step(at_failed)]
                    elif delivery.is_pulling:
                        # Client want to pull data from server, we need to upload
                        [step(after_upload, self.transfer(to=True)) if step(
                            before_upload) else step(at_failed)]
                    else:
                        step(at_invalid_state)
                    # update regularly.
                    step(at_last)
                    # sleep a short time is very import!
                    # otherwise, some states may be rewrite.
                    time.sleep(0.1)

            try_times = 0 if cnt else try_times + 1

            if cnt == 0:
                logger.info(
                    f"Empty delivery, waiting {try_times}/{self.max_try_times}...")
                time.sleep(5.0)

            if try_times >= self.max_try_times:
                self.manual_stop()

            # left some time to maintainer lock
            time.sleep(0.1)
        return "Backend exited."

    def backend_loop(self) -> bool:
        """If go into run() func, will return True, otherwise False.
        """
        return self.run() if self.role == leader else False

    def finish(self, auto_exit: bool = False):
        if self.maintainer:
            self.maintainer.manual_stop()
            world = self.maintainer.world
            for delivery in Delivery.delivery_generator():
                if delivery.world == world:
                    Destroy.destroy_delivery(delivery)

        if auto_exit and self.role == leader:
            exit(15)

    def __getattribute__(self, name: str) -> Any:
        """Try to fetch the attribute of api. If failed, try to fetch it from delivery.
        """
        if name == 'delivery':
            return super().__getattribute__(name)

        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            return getattr(self.delivery, name)

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="OpenFedAPI",
            description=f"ROLE: {self.role}"
        )

    def __enter__(self):
        self.original_state = [
            openfed.DAL.is_dal,
            openfed.ASYNC_OP.is_async_op,
        ]

        openfed.DAL.set(self.dal)
        openfed.ASYNC_OP.set(self.async_op)
        peeper.api_lock.acquire()
        peeper.api = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore state
        dal, async_op = self.original_state

        openfed.DAL.set(dal)
        openfed.ASYNC_OP.set(async_op)
        peeper.api = None
        peeper.api_lock.release()
