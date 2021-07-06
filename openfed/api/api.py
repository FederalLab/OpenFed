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
from typing import Any, Callable, Dict, List, Union

import openfed
from openfed.common import (Address_, Hook, SafeThread, TaskInfo,
                            default_address, logger)
from openfed.container import Agg, Reducer
from openfed.core import (Collector, Cypher, Destroy, Maintainer, Reign, World,
                          openfed_lock)
from openfed.core.utils import DeviceOffline
from openfed.core.utils.lock import del_maintainer_lock
from openfed.pipe import Pipe
from openfed.utils import (convert_to_list, keyboard_interrupt_handle,
                           openfed_class_fmt)
from torch import Tensor
from torch.optim import Optimizer

from .step import (Step, after_destroy, after_download, after_upload,
                   at_failed, at_first, at_invalid_state, at_last,
                   at_new_episode, at_zombie, before_destroy, before_download,
                   before_upload)


def _device_offline_care(func):
    def device_offline_care(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except DeviceOffline as e:
            logger.error("Device offline.")
            return False
    return device_offline_care


class API(SafeThread, Hook):
    """Provide a unified api for backend and frontend.
    """

    # Communication related
    maintainer: Maintainer
    reign: Reign

    def __init__(self,
                 frontend: bool = True,
                 dal: bool = True):
        """Whether act as a frontend.
        Frontend is always in sync mode, which will ease the coding burden.
        Backend will be set as async mode by default.
        """
        # Call SafeThread init function.
        SafeThread.__init__(self, daemon=True)
        Hook.__init__(self)

        self.dal: bool = dal
        self.frontend: bool = frontend

        # Set a flag for backend.
        self.backend: bool = not self.frontend
        keyboard_interrupt_handle()

        # Enable async_op if this is backend.
        self.async_op: bool = self.backend

        # Set default value
        self.version: int = 0

        self._hooks_del: List[Cypher] = []
        self._hooks_inf: List[Collector] = []

        self.stopped: bool = False
        self.received_numbers: int = 0
        self.last_aggregate_time: float = time.time()
        self.reign_task_info: TaskInfo = TaskInfo()
        self.task_info_list: List[TaskInfo] = []

        # Data handle
        self.state_dict: Dict[str, Tensor] = {}
        self.aggregator: List[Agg] = []
        self.optimizer: List[Optimizer] = []
        self.ft_optimizer: List[Optimizer] = []
        self.reducer: List[Reducer] = []

        # how many times for backend waiting for connections.
        self.max_try_times: int = 5

        # Set them here to avoid get attribute error.
        self.reign = None
        self.maintainer = None

    def add_informer_hook(self, hook: Collector):
        self._hooks_inf.append(hook)

    def add_deliver_hook(self, hook: Cypher):
        self._hooks_del.append(hook)

    def _add_hook_to_reign(self):
        # register a clone of informer hook.
        # informer hook may contain some inner variable, which is not allowed
        # to share with each other.
        [self.reign.register_collector(hook.clone(
        )) for hook in self._hooks_inf if hook.bounding_name not in self.reign._hook_dict]
        # register the hook directly.
        # deliver hook is not allowed to have inner parameters.
        # it can be used among all reign.
        [self.reign.register_cypher(
            hook) for hook in self._hooks_del if hook not in self.reign._hook_list]

    def build_connection(self, world: World = None, address: Union[Address_, List[Address_]] = None, address_file: str = None):
        world = world if world is not None else World(leader=self.backend)
        # Check identity
        assert world.leader == self.backend
        assert world.follower == self.frontend

        if address is None and address_file is None:
            address = default_address

        if self.backend and openfed.DAL.is_dal:
            # NOTE: hold openfed_lock before create a dynamic address loading maintainer.
            # otherwise, it may interrupt the process and cause error before you go into loop()
            openfed_lock.acquire()
        self.maintainer = Maintainer(
            world, address=address, address_file=address_file)

        if self.frontend:
            self.reign = Reign.default_reign()

    def init_reign(self):
        """Init a reign.
        First, register hook to reign, including hook for informer and deliver.
        Second, apply collect and scatter hook.
        Third, set self task info and load others task info
        Last, set the state dict.

        """
        # 1. register hook first
        self._add_hook_to_reign()

        # 2. gather hook information
        self.reign.collect()
        self.reign.scatter()

        # 4. set state dict
        assert self.state_dict
        self.reign.reset_state_dict(self.state_dict)

    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        self.state_dict = state_dict

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="OpenFedAPI",
            description=f"{'Frontend' if self.frontend else 'Backend'}."
        )

    def finish(self, auto_exit: bool = False):
        if self.maintainer:
            Destroy.destroy_all_in_a_world(self.maintainer.world)
            self.maintainer.manual_stop()
            del_maintainer_lock(self.maintainer)

        if auto_exit and self.backend:
            exit(15)

    def backend_loop(self) -> bool:
        """If go into run() func, will return True, otherwise False.
        """
        return self.run() if self.backend else False

    def __enter__(self):
        self.original_state = [
            openfed.DAL.is_dal,
            openfed.ASYNC_OP.is_async_op,
        ]

        openfed.DAL.set(self.dal)
        openfed.ASYNC_OP.set(self.async_op)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore state
        dal, async_op = self.original_state

        openfed.DAL.set(dal)
        openfed.ASYNC_OP.set(async_op)

    def set_task_info(self, task_info: TaskInfo) -> None:
        self.reign_task_info = task_info

    def get_task_info(self) -> TaskInfo:
        return self.reign_task_info

    def update_version(self, version: int = None):
        """Update inner model version.
        """
        self.version = version if version is not None else self.version + 1

    @_device_offline_care
    def upload(self) -> bool:
        """As for frontend, it is much easier for us to judge the new version.
        A download and upload is build a version updating.
        So increase version number here.
        """
        self.init_reign()

        self.reign.set_task_info(self.reign_task_info)
        # unpack state
        [self.pack_state(ft_opt) for ft_opt in self.ft_optimizer]

        flag = self.reign.upload(self.version)
        return flag if flag or self.backend else self._wait_handler()

    @_device_offline_care
    def download(self) -> bool:
        """In frontend, the frontend optimizer state dict will automatically unpack after download.
        But, in backend, it won't. You should deal with this in hook step.
        """
        self.init_reign()
        flag = self.reign.download(self.version)

        def callback():
            if self.frontend:
                # unpack state
                [self.unpack_state(ft_opt)
                 for ft_opt in self.ft_optimizer]
                self.version = self.reign.upload_version
                self.reign_task_info = self.reign.task_info
        if flag:
            callback()

        return flag if flag or self.backend else self._wait_handler(callback)

    def _wait_handler(self, callback: Callable = lambda: ...):
        if openfed.ASYNC_OP.is_async_op:
            while not self.reign.deal_with_hang_up():
                if self.reign.is_offline:
                    return False
                time.sleep(0.1)
            else:
                callback()
                return True
        else:
            return False

    def set_aggregator_and_optimizer(self,
                                     aggregator: Union[Agg, List[Agg]],
                                     optimizer: Union[Optimizer, List[Optimizer]],
                                     ft_optimizer: Union[Optimizer, List[Optimizer]] = None):
        """
        Args:
            We will set the same optimizer for frontend and backend.
            If you have specified different optimizer for frontend, we will use it.

            Actually, you can put the pipe_optimizer to ft_optimizer, it will automatically 
            pack the state before upload and unpack state after download.
        """
        self.aggregator = convert_to_list(aggregator)
        self.optimizer = convert_to_list(optimizer)
        if ft_optimizer is not None:
            self.ft_optimizer = convert_to_list(ft_optimizer)
        else:
            self.ft_optimizer = convert_to_list(optimizer)

        assert len(self.aggregator) == len(
            self.optimizer), "Aggregator must be corresponding to Optimizer."

    def set_aggregator(self, aggregator: Union[Agg, List[Agg]]):
        pass

    def set_ft_optimizer(self, optimizer: Union[Optimizer, List[Optimizer]]):
        pass

    def set_bk_optimizer(self, optimizer: Union[Optimizer, List[Optimizer]]):
        pass

    def set_pipe(self, pipe: Union[Pipe, List[Pipe]]):
        pass

    def set_reducer(self, reducer: Union[Reducer, List[Reducer]]) -> None:
        """
        Args: 
            reducer: used to aggregate task info.
        """
        self.reducer.extend(convert_to_list(reducer))

    def safe_run(self):
        """
            Use self.run() to start this loop in the main thread.
            Use self.start() to start this loop in the thread.
        """
        if self.frontend:
            return None
        # NOTE: release openfed_lock here.
        if openfed.DAL.is_dal:
            openfed_lock.release()

        try_times = 0
        while not self.stopped:
            with self.maintainer.mt_lock:
                self.step(at_new_episode)
                rg = Reign.reign_generator()
                cnt = 0
                for reign in rg:
                    if self.stopped or reign is None:
                        break
                    # assign reign to self first.
                    self.reign = reign

                    # register hook to reign if necessary.
                    self._add_hook_to_reign()

                    cnt += 1
                    self.step(at_first)
                    if reign.is_offline:
                        [self.step(after_destroy, Destroy.destroy_reign(reign)) if self.step(
                            before_destroy) else self.step(at_failed)]
                    elif reign.upload_hang_up:
                        self.step(after_upload,
                                  reign.deal_with_hang_up())
                    elif reign.download_hang_up:
                        self.step(after_download,
                                  reign.deal_with_hang_up())
                    elif reign.is_zombie:
                        self.step(at_zombie)
                    elif reign.is_pushing:
                        # Client want to push data to server, we need to download.
                        [self.step(after_download, self.download()) if self.step(
                            before_download) else self.step(at_failed)]
                    elif reign.is_pulling:
                        # Client want to pull data from server, we need to upload
                        [self.step(after_upload, self.upload()) if self.step(
                            before_upload) else self.step(at_failed)]
                    else:
                        self.step(at_invalid_state)
                    # update regularly.
                    self.step(at_last)

            try_times = 0 if cnt else try_times + 1

            if cnt == 0:
                logger.info(
                    f"Empty reign, waiting {try_times}/{self.max_try_times}...")
                time.sleep(5.0)

            if try_times >= self.max_try_times:
                self.manual_stop()

            # left some time to maintainer lock
            time.sleep(0.1)
        return "Backend exited."

    def register_step(self, step: Step):
        """Register the step function to step possition call."""
        for name in convert_to_list(step.step_name):
            cnt = 0
            for n in self.hook_dict.keys():
                if n.startswith(name):
                    cnt = max(cnt, int(n.split(".")[-1]))
            else:
                self.register_hook(f"{name}.{cnt+1}", step)

    def step(self, step_name: str, *args, **kwargs) -> Union[None, bool]:
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

    def __getattribute__(self, name: str) -> Any:
        """Try to fetch the attribute of api. If failed, try to fetch it from reign.
        """
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            return getattr(self.reign, name)
