import time
from typing import Dict, List, Union

import openfed
from openfed.aggregate import Aggregator
from openfed.common import (MAX_TRY_TIMES, Address, Hook, Peeper, SafeTread,
                            default_address)
from openfed.common.logging import logger
from openfed.federated import Destroy, Maintainer, Reign, World
from openfed.federated.utils.lock import openfed_lock
from openfed.unified.step.after_destroy import AfterDestroy
from openfed.unified.step.after_download import AfterDownload
from openfed.unified.step.after_upload import AfterUpload
from openfed.unified.step.at_failed import AtFailed
from openfed.unified.step.at_invalid_state import AtInvalidState
from openfed.unified.step.at_new_episode import AtNewEpisode
from openfed.unified.step.at_zombie import AtZombie
from openfed.unified.step.base import Step
from openfed.unified.step.before_destroy import BeforeDestroy
from openfed.unified.step.before_download import BeforeDownload
from openfed.unified.step.before_upload import BeforeUpload
from openfed.unified.unify import Unify
from openfed.unified.utils import _backend_access, _convert_to_list
from openfed.utils import openfed_class_fmt
from torch import Tensor
from torch.optim import Optimizer


class Backend(Unify, SafeTread, Peeper, Hook):
    """An unified API of backend for users.
    """
    aggregator: List[Aggregator]
    optimizer: List[Optimizer]

    state_dict: List[Dict[str, Tensor]]

    task_info_list: List[Dict]

    maintainer: Maintainer

    reign: Reign

    version: int

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

    @_backend_access
    def build_connection(self,
                         world: World = None,
                         address: Union[Address, List[Address]] = None,
                         address_file: str = None):
        """Build a World with address and address file. 
        initialize self at the same time.
        """
        if world is None:
            world = World(king=True)
        else:
            assert world.king, "Backend must be king."

        self.world = world

        if address is None and address_file is None:
            address = default_address

        # NOTE: hold openfed_lock before create a dynamic address loading maintainer.
        # otherwise, it may interrupt the process and cause error before you go into loop()
        if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
            openfed_lock.acquire()

        self.maintainer = Maintainer(
            world, address=address, address_file=address_file)

    @_backend_access
    def set_aggregate_triggers(self, trigger: Step):
        self.register_step(trigger)
        self.aggregate_triggers = True

    @_backend_access
    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        logger.info(
            f"{'Set' if not self.state_dict else 'Unset'} state dict.")
        self.state_dict = state_dict

    @_backend_access
    def set_aggregator_and_optimizer(self, aggregator: Union[Aggregator, List[Aggregator]], optimizer: Union[Optimizer, List[Optimizer]]):
        aggregator = _convert_to_list(aggregator)
        optimizer = _convert_to_list(optimizer)
        assert len(aggregator) == len(optimizer)
        self.aggregator = aggregator
        self.optimizer = optimizer

    @_backend_access
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
                self.step("at_new_episode")
                rg = Reign.reign_generator()
                cnt = 0
                self.loop_times += 1
                for reign in rg:
                    if not self.stopped and reign is not None:
                        cnt += 1
                        self.reign = reign
                        self.step("at_first")
                        if reign.is_offline:
                            # Destroy process
                            if self.step("before_destroy"):
                                self.step("after_destroy",
                                          Destroy.destroy_reign(reign))
                            else:
                                self.step("at_failed")
                        elif reign.upload_hang_up:
                            self.step("after_upload",
                                      reign.deal_with_hang_up())
                        elif reign.download_hang_up:
                            self.step("after_download",
                                      reign.deal_with_hang_up())
                        elif reign.is_zombie:
                            self.step("at_zombie")
                        elif reign.is_pushing:
                            # Client want to push data to server, we need to download.
                            if self.step("before_download"):
                                self.step("after_download",
                                          reign.download(self.version))
                            else:
                                self.step("at_failed")
                        elif reign.is_pulling:
                            # Client want to pull data from server, we need to upload
                            # if self.step_before_upload():
                            if self.step("before_upload"):
                                self.step("after_upload",
                                          reign.upload(self.version))
                            else:
                                self.step("at_failed")
                        else:
                            self.step("at_invalid_state")
                    # update regularly.
                    self.step("at_last")
                else:
                    del rg
            if cnt == 0:
                max_try_times += 1
                logger.warning(
                    f"Max Try Times: {max_try_times}/{MAX_TRY_TIMES}")
                logger.warning(f"Empty Reign\n{self}")
                time.sleep(openfed.SLEEP_LONG_TIME.seconds)
            else:
                max_try_times = 0

            if max_try_times >= MAX_TRY_TIMES:
                self.manual_stop()

            # left some time to maintainer lock
            time.sleep(openfed.SLEEP_SHORT_TIME.seconds)
        self.finish()
        return "Backend exited."

    @_backend_access
    def register_step(self, step: Step):
        name = step.step_name
        name = f"{name}.{len(self.hook_dict)}"
        self.register_hook(key=name, func=step)

    @_backend_access
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
                output.append(hook(self, *args, **kwargs))
        if not output:
            return None

        # reduce output
        if None in output:
            return None

        if False in output:
            return False

        return True

    @_backend_access
    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Backend",
            description=str(self.maintainer)
        )
