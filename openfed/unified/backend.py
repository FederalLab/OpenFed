import time
from typing import Dict, List, Union

import openfed
from openfed.aggregate import Aggregator
from openfed.common import (MAX_TRY_TIMES, Address, Hook, SafeTread, TaskInfo,
                            default_address, logger)
from openfed.federated import Destroy, Maintainer, Reign, World, openfed_lock
from openfed.utils import openfed_class_fmt
from torch import Tensor
from torch.optim import Optimizer

from .step import (AfterDestroy, AfterDownload, AfterUpload, AtFailed, AtFirst,
                   AtInvalidState, AtNewEpisode, AtZombie, BeforeDestroy,
                   BeforeDownload, BeforeUpload, Step)
from .unify import Unify
from .utils import (after_connection, after_destroy, after_download,
                    after_upload, at_failed, at_first, at_invalid_state,
                    at_last, at_new_episode, at_zombie, backend_access,
                    before_destroy, before_download, before_upload,
                    convert_to_list)


class Backend(Unify, SafeTread, Hook):
    """An unified API of backend for users.
    """
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

    def __init__(self, *args,  **kwargs):
        Unify.__init__(self, *args, **kwargs)
        SafeTread.__init__(self)
        register_default_step_for_backend = kwargs['register_default_step_for_backend']

        # Set default value.
        self.stopped = False
        self.version = 0
        self.reign = None
        self.received_numbers = 0
        self.last_aggregate_time = time.time()

        # Initialize properties
        self.aggregator = None
        self.optimizer = None
        self.state_dict = None
        self.maintainer = None
        self.reign = None

        if register_default_step_for_backend:
            self.register_step(AfterDestroy())
            self.register_step(AfterDownload())
            self.register_step(AfterUpload())
            self.register_step(AtFirst())
            self.register_step(AtFailed())
            self.register_step(AtInvalidState())
            # There may be some different operations on at last operations.
            # Do not register it as a default operation.
            # self.register_step(AtLast())
            self.register_step(AtNewEpisode())
            self.register_step(AtZombie())
            self.register_step(BeforeDestroy())
            self.register_step(BeforeDownload())
            self.register_step(BeforeUpload())

    @backend_access
    def build_connection(self,
                         world: World = None,
                         address: Union[Address, List[Address]] = None,
                         address_file: str = None):
        """Build a World with address and address file. 
        initialize self at the same time.
        """
        if world is None:
            world = World(leader=True)
        else:
            assert world.leader, "Backend must be leader."

        self.world = world

        if address is None and address_file is None:
            address = default_address

        # NOTE: hold openfed_lock before create a dynamic address loading maintainer.
        # otherwise, it may interrupt the process and cause error before you go into loop()
        if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
            openfed_lock.acquire()

        self.maintainer = Maintainer(
            world, address=address, address_file=address_file)

    @backend_access
    def set_aggregate_triggers(self, trigger: Step):
        self.register_step(trigger)
        self.aggregate_triggers = True

    @backend_access
    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        logger.debug(
            f"{'Set' if not self.state_dict else 'Unset'} state dict.")
        self.state_dict = state_dict

    @backend_access
    def set_aggregator_and_optimizer(self, aggregator: Union[Aggregator, List[Aggregator]], optimizer: Union[Optimizer, List[Optimizer]]):
        aggregator = convert_to_list(aggregator)
        optimizer = convert_to_list(optimizer)
        assert len(aggregator) == len(optimizer)
        self.aggregator = aggregator
        self.optimizer = optimizer

    @backend_access
    @after_connection
    def safe_run(self):
        """
            Use self.run() to start this loop in the main thread.
            Use self.start() to start this loop in the thread.
        """
        # NOTE: release openfed_lock here.
        if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
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

    @backend_access
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

    @backend_access
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
            del_keys = [key for key in self.hook_dict if key.startswith(step_name)]
            for d_key in del_keys:
                del self.hook_dict[d_key]
        self.register_step(step)

    @backend_access
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
        if not output:
            return None

        # reduce output
        if None in output:
            return None

        if False in output:
            return False

        return True

    @backend_access
    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Backend",
            description=str(self.maintainer)
        )
