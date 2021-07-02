import time
from typing import Any, Callable, Dict, List, Union

import openfed
from openfed.aggregate import Aggregator
from openfed.common import (MAX_TRY_TIMES, Address, Hook, SafeTread, TaskInfo,
                            default_address, logger)
from openfed.federated import (Destroy, Maintainer, Peeper, Reign, World,
                               openfed_lock)
from openfed.utils import keyboard_interrupt_handle, openfed_class_fmt
from torch import Tensor
from torch.optim import Optimizer

from .after import AfterDownload
from .before import BeforeUpload
from .step import (Step, after_destroy, after_download, after_upload,
                   at_failed, at_first, at_invalid_state, at_last,
                   at_new_episode, at_zombie, before_destroy, before_download,
                   before_upload)


def convert_to_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


class API(SafeTread, Hook, Peeper):
    """Provide a unified api for backend and frontend.
    """
    maintainer: Maintainer = None

    reign: Reign = None

    world: World = None

    version: int = 0

    frontend: bool = True
    # fontend xor backward == True
    backend: bool = False

    # backend is True default, frontend is False default.
    async_op: bool = False

    dal: bool = True

    state_dict: List[Dict[str, Tensor]] = None

    _hooks_for_informers: List[Callable] = None
    _hooks_for_delivers: List[Callable] = None

    aggregator: List[Aggregator]
    optimizer: List[Optimizer]

    # A List to record all task info aggregated by this backend
    task_info_list: List[TaskInfo]

    # A dictionary to record the task info message of current reign.
    reign_task_info: TaskInfo

    loop_times: int

    received_numbers: int

    # A flag to indicate whether set the triggered step for backend.
    aggregate_triggers: bool

    def __init__(self,
                 frontend: bool = True,
                 dal: bool = True):
        """Whether act as a frontend.
        Frontend is always in sync mode, which will ease the coding burden.
        Backend will be set as async mode by default.
        """
        # Call SafeThread init function.
        super().__init__(self)

        self.frontend = frontend
        # Set a flag for backend.
        self.backend = not self.frontend
        keyboard_interrupt_handle()

        self.async_op = self.backend
        self.dal = dal

        # Set default value
        self.version = 0

        self._hooks_for_delivers = []
        self._hooks_for_informers = []

        # Set default value.
        self.stopped = False
        self.received_numbers = 0
        self.last_aggregate_time = time.time()

        # Initialize properties
        self.aggregator = None
        self.optimizer = None
        self.state_dict = None
        self.maintainer = None
        self.reign = None
        self.reign_task_info = TaskInfo()

        self.register_step(AfterDownload())
        self.register_step(BeforeUpload())

    def add_informer_hook(self, hook: Callable):
        self._hooks_for_informers.append(hook)

    def add_deliver_hook(self, hook: Callable):
        self._hooks_for_delivers.append(hook)

    def _add_hook_to_reign(self):
        for hook in self._hooks_for_informers:
            if hook.bounding_name not in self.reign._hook_dict:
                # register a clone of informer hook.
                # informer hook may contain some inner variable, which is not allowed
                # to share with each other.
                self.reign.register_collector(hook.clone())
        for hook in self._hooks_for_delivers:
            if hook not in self.reign._hook_list:
                # register the hook directly.
                # deliver hook is not allowed to have inner parameters.
                # it can be used among all reign.
                self.reign.register_cypher(hook)

    @property
    def nick_name(self) -> str:
        return self.reign.nick_name

    def build_connection(self, world: World = None, address: Union[Address, List[Address]] = None, address_file: str = None):
        world = world if world is not None else World(leader=self.backend)
        # Check identity
        assert world.leader == self.backend
        assert world.follower == self.frontend
        self.world = world

        address = default_address if address is None and address_file is None else address

        if self.backend and openfed.DAL.is_dal:
            # NOTE: hold openfed_lock before create a dynamic address loading maintainer.
            # otherwise, it may interrupt the process and cause error before you go into loop()
            openfed_lock.acquire()
        self.maintainer = Maintainer(
            world, address=address, address_file=address_file)

        self.reign = Reign.default_reign() if self.frontend else None

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

        # 3. task info
        self.reign.set_task_info(self.reign_task_info)
        self.reign_task_info = self.reign.task_info

        # 4. set state dict
        assert self.state_dict
        self.reign.reset_state_dict(self.state_dict)

    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        self.state_dict = state_dict

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="OpenFedAPI",
            description=f"{'Frontend' if self.frontend else 'Backend'}."
        )

    def finish(self, auto_exit: bool = True):
        Destroy.destroy_all_in_a_world(self.world)
        if self.maintainer:
            self.maintainer.manual_stop()

        if auto_exit and self.backend:
            exit(0)

    # @backend_access
    # @after_connection
    # def run(self, *args, **kwargs):
    #     return SafeTread.run(self, *args, **kwargs)

    def backend_loop(self):
        return self.run() if self.backend else None

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

    def set(self, key: str, value: Any) -> None:
        self.reign.set(key, value)

    def get(self, key: str) -> Any:
        return self.reign.get(key)

    def upload(self) -> bool:
        """As for frontend, it is much easier for us to judge the new version.
        A download and upload is build a version updating.
        So increase version number here.
        """
        self.init_reign()

        # unpack state
        for ft_opt in self.frontend_optimizer:
            self.pack_state(ft_opt)

        flag = self.reign.upload(self.version)
        return flag if flag or self.backend else self._wait_handler()

    def download(self) -> bool:
        """In frontend, the frontend optimizer state dict will automatically unpack after download.
        But, in backend, it won't. You should deal with this in hook step.
        """
        self.init_reign()
        flag = self.reign.download(self.version)

        def callback():
            if self.frontend:
                # unpack state
                for ft_opt in self.frontend_optimizer:
                    self.unpack_state(ft_opt)

        if flag:
            callback()

        return flag if flag or self.backend else self._wait_handler(callback)

    def _wait_handler(self, callback: Callable = lambda: ...):
        if openfed.ASYNC_OP.is_async_op:
            while not self.reign.deal_with_hang_up():
                if self.reign.is_offline:
                    return False
                time.sleep(openfed.SLEEP_SHORT_TIME.seconds)
            callback()
            return True
        else:
            return False

    def pack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.pack_state(obj, keys)

    def unpack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.unpack_state(obj, keys)

    def set_aggregate_triggers(self, trigger: Step):
        self.register_step(trigger)
        self.aggregate_triggers = True

    def set_aggregator_and_optimizer(self,
                                     aggregator: Union[Aggregator, List[Aggregator]],
                                     optimizer: Union[Optimizer, List[Optimizer]],
                                     frontend_optimizer: Union[Optimizer, List[Optimizer]] = None):
        """
        Args:
            We will set the same optimizer for frontend and backend.
            If you have specified different optimizer for frontend, we will use it.

            Actually, you can put the pipe_optimizer to frontend_optimizer, it will automatically 
            pack the state before upload and unpack state after download.
        """
        aggregator = convert_to_list(aggregator)
        optimizer = convert_to_list(optimizer)
        frontend_optimizer = frontend_optimizer if frontend_optimizer is not None else optimizer

        assert len(aggregator) == len(optimizer)
        self.aggregator = aggregator
        self.optimizer = optimizer

        self.frontend_optimizer = frontend_optimizer

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

        assert self.aggregate_triggers, "Call `self.set_aggregate_triggers()` first."

        max_try_times = 0
        self.loop_times = 0
        while not self.stopped:
            with self.maintainer.maintainer_lock:
                self.step(at_new_episode)
                rg = Reign.reign_generator()
                cnt = 0
                self.loop_times += 1
                for reign in rg:
                    if not self.stopped and reign is not None:
                        # assign reign to self first.
                        self.reign = reign

                        # register hook to reign if necessary.
                        self._add_hook_to_reign()

                        cnt += 1
                        self.step(at_first)
                        if reign.is_offline:
                            # Destroy process
                            if self.step(before_destroy):
                                self.step(after_destroy,
                                          Destroy.destroy_reign(reign))
                            else:
                                self.step(at_failed)
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
                            if self.step(before_download):
                                self.step(after_download,
                                          self.download())
                            else:
                                self.step(at_failed)
                        elif reign.is_pulling:
                            # Client want to pull data from server, we need to upload
                            # if self.step_before_upload():
                            if self.step(before_upload):
                                self.step(after_upload,
                                          self.upload())
                            else:
                                self.step(at_failed)
                        else:
                            self.step(at_invalid_state)
                    # update regularly.
                    self.step(at_last)
                else:
                    del rg
            if cnt == 0:
                max_try_times += 1
                logger.debug(
                    f"Max Try Times: {max_try_times}/{MAX_TRY_TIMES}")
                logger.debug(f"Empty Reign\n{self}")
                time.sleep(openfed.SLEEP_LONG_TIME.seconds)
            else:
                max_try_times = 0

            if max_try_times >= MAX_TRY_TIMES:
                self.manual_stop()

            # left some time to maintainer lock
            time.sleep(openfed.SLEEP_SHORT_TIME.seconds)
        self.finish()
        return "Backend exited."

    def register_step(self, step: Step):
        """Register the step function to step possition call."""
        names = step.step_name
        if isinstance(names, str):
            names = [names]

        for name in names:
            cnt = 0
            for n in self.hook_dict.keys():
                if n.startswith(name):
                    cnt = max(cnt, int(n.split(".")[-1]))
            name = f"{name}.{cnt+1}"
            self.register_hook(key=name, func=step)

    def replace_step(self, step_name: str, step: Step):
        """Replace the already registered step function at this step with the new one.
        NOTE: Be careful to call this function! If step_name is 'XXX.cnt' formot, we will delete it first and then add this new one.
        If step_name is 'XXX', we will first remove all the step function and add this new one.
        NOTE: if step is a multi-step function, this function will register it to other step at the some time.

        This function is useful if you want to replace the default BeforeUpload with Dispatch function.
        """
        if len(step_name.split('.')) == 2:
            assert step_name in self.hook_dict
            del self.hook_dict[step_name]
        else:
            del_keys = [
                key for key in self.hook_dict if key.startswith(step_name)]
            for d_key in del_keys:
                del self.hook_dict[d_key]
        self.register_step(step)

    def step(self, step_name: str, *args, **kwargs) -> Union[None, bool]:
        """
            You can chain the same type hook together.
            Hook will return a bool value or None.
            If bool is returned, we will use `and` to reduce them.
            If None is returned, we will return `None` directly.
            You should directly store other variables in self object.
        """
        output = []
        for name, hook in self.hook_dict.items():
            if name.startswith(step_name):
                hook.current_step = step_name
                output.append(hook(self, *args, **kwargs))

        # reduce output
        if None in output:
            return None

        if False in output:
            return False

        return True
