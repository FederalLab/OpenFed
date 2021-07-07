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

import openfed
from openfed.common import (Address_, Hook, SafeThread, TaskInfo,
                            default_address, logger)
from openfed.common.base.peeper import peeper
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

from .functional import download_callback
from .step import (Step, after_destroy, after_download, after_upload,
                   at_failed, at_first, at_invalid_state, at_last,
                   at_new_episode, at_zombie, before_destroy, before_download,
                   before_upload)

peeper.api_lock = Lock()


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
    current_step: str

    def __init__(self,
                 frontend: bool,
                 state_dict: Dict[str, Tensor],
                 ft_optimizer: Union[Optimizer, List[Optimizer]],
                 aggregator: Union[Agg, List[Agg]],
                 bk_optimizer: Union[Optimizer, List[Optimizer]]=None,
                 pipe: Union[Pipe, List[Pipe]] = None,
                 reducer: Union[Reducer, List[Reducer]] = None,
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
        self.state_dict: Dict[str, Tensor] = state_dict
        self.aggregator: List[Agg] = convert_to_list(aggregator)
        if bk_optimizer is None:
            bk_optimizer = ft_optimizer
        self.pipe: List[Pipe] = convert_to_list(pipe)
        self.bk_optimizer: List[Optimizer] = convert_to_list(bk_optimizer)
        self.ft_optimizer: List[Optimizer] = convert_to_list(ft_optimizer)
        self.reducer: List[Reducer] = convert_to_list(reducer)

        assert len(self.aggregator) == len(self.bk_optimizer) == len(self.ft_optimizer)

        # how many times for backend waiting for connections.
        self.max_try_times: int = 5

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
            self._hooks_inf.append(hook)
        elif isinstance(hook, Cypher):
            self._hooks_del.append(hook)
        else:
            raise NotImplementedError(
                f'Hook type is not supported: {type(hook)}.')

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
            self._add_hook_to_reign()

    def update_version(self, version: int = None):
        """Update inner model version.
        """
        self.version = version if version is not None else self.version + 1

    @_device_offline_care
    def transfer(self, to: bool = False, task_info: TaskInfo = None) -> bool:
        r"""Transfer inner state to other ends.
        Args:
            to: if true, upload data to the other end.
                if false, download data from the other end.
            task_info: task info want to upload or download.
                The new task info will be updated directly to this parameters.
        """
        # 1. gather hook information
        self.reign.collect()
        self.reign.scatter()

        # 2. set state dict
        assert self.state_dict
        self.reign.reset_state_dict(self.state_dict)

        if to:
            # 3. set task info
            if task_info is not None:
                self.reign.set_task_info(task_info)
                self.reign_task_info = task_info
            else:
                self.reign.set_task_info(self.reign_task_info)

            # 4. Pack state
            if self.frontend:
                [self.pack_state(ft_opt) for ft_opt in self.ft_optimizer]
            elif self.backend:
                [self.pack_state(bk_opt) for bk_opt in self.bk_optimizer]
            if self.pipe is not None:
                [self.pack_state(pipe) for pipe in self.pipe]

            # 5. transfer
            flag = self.reign.upload(self.version)
        else:
            flag = self.reign.download(self.version)

        def callback():
            if not to:
                download_callback(self)
                if task_info is not None:
                    task_info.load_dict(self.reign_task_info.info_dict)

        # 7. hand on
        if flag:
            callback()
            return True
        else:
            if self.backend:
                # return directly.
                return flag
            else:
                # wait for finished.
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

    def safe_run(self):
        """
            Use self.run() to start this loop in the main thread.
            Use self.start() to start this loop in the thread.
        """

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

        if self.frontend:
            return None
        # NOTE: release openfed_lock here.
        if openfed.DAL.is_dal:
            openfed_lock.release()

        try_times = 0
        while not self.stopped:
            with self.maintainer.mt_lock:
                step(at_new_episode)
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
                    step(at_first)
                    if reign.is_offline:
                        [step(after_destroy, Destroy.destroy_reign(reign)) if step(
                            before_destroy) else step(at_failed)]
                    elif reign.upload_hang_up:
                        step(after_upload,
                             reign.deal_with_hang_up())
                    elif reign.download_hang_up:
                        step(after_download,
                             reign.deal_with_hang_up())
                    elif reign.is_zombie:
                        step(at_zombie)
                    elif reign.is_pushing:
                        # Client want to push data to server, we need to download.
                        [step(after_download, self.transfer(to=False)) if step(
                            before_download) else step(at_failed)]
                    elif reign.is_pulling:
                        # Client want to pull data from server, we need to upload
                        [step(after_upload,self.transfer(to=True)) if step(
                            before_upload) else step(at_failed)]
                    else:
                        step(at_invalid_state)
                    # update regularly.
                    step(at_last)

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

    def backend_loop(self) -> bool:
        """If go into run() func, will return True, otherwise False.
        """
        return self.run() if self.backend else False

    def finish(self, auto_exit: bool = False):
        if self.maintainer:
            Destroy.destroy_all_in_a_world(self.maintainer.world)
            self.maintainer.manual_stop()
            del_maintainer_lock(self.maintainer)

        if auto_exit and self.backend:
            exit(15)

    def __getattribute__(self, name: str) -> Any:
        """Try to fetch the attribute of api. If failed, try to fetch it from reign.
        """
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            return getattr(self.reign, name)

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="OpenFedAPI",
            description=f"{'Frontend' if self.frontend else 'Backend'}."
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

