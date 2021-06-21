import time
from typing import Dict, List, Union, overload

from loguru import logger
from torch import Tensor
from torch.optim import Optimizer

import openfed
from openfed.aggregate import Aggregator
from openfed.common import Address, Hook, Peeper, SafeTread, default_address
from openfed.common.unify import Unify, _backend_access
from openfed.federated import Destroy, Maintainer, Reign, World, openfed_lock
from openfed.utils import openfed_class_fmt


class Backend(Unify, SafeTread, Peeper, Hook):
    """An unified API of backend for users.
    """
    aggregator: Aggregator
    optimizer: Optimizer

    state_dict: Dict[str, Tensor]

    maintiner: Maintainer

    reign: Reign

    version: int

    received_numbers: int

    last_aggregate_time: float

    frontend: bool = False

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
        # otherwise, it may interrupte the process and cause error before you go into loop()
        if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
            openfed_lock.acquire()

        self.maintainer = Maintainer(
            world, address=address, address_file=address_file)

        # Set default value.
        self.stopped = False
        self.version = 0
        self.reign = None
        self.received_numbers = 0
        self.last_aggregate_time = time.time()

        SafeTread.__init__(self)

    @_backend_access
    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        self.state_dict = state_dict

    @_backend_access
    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    @_backend_access
    def set_aggregator(self, aggregator: Aggregator):
        self.aggregator = aggregator

    @_backend_access
    def safe_run(self):
        """
            Use self.run() to start this loop in the main thread.
            Use self.start() to start this loop in the thread.
        """
        # NOTE: release openfed_lock here.
        if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
            openfed_lock.release()
        while not self.stopped:
            with self.maintainer.maintainer_lock:
                self.step_at_new_episode()
                rg = Reign.reign_generator()
                for reign in rg:
                    if not self.stopped and reign is not None:
                        self.reign = reign
                        self.step_at_first()
                        if reign.is_zombine:
                            self.step_at_zombine()
                        elif reign.is_offline:
                            # Destroy process
                            if self.step_before_destroy():
                                self.step_after_destroy(
                                    Destroy.destroy_reign(reign))
                            else:
                                self.step_at_failed()
                        elif reign.is_pushing:
                            # Client want to push data to server, we need to download.
                            if self.step_before_download():
                                self.step_after_download(reign.download())
                            else:
                                self.step_at_failed()
                        elif reign.is_pulling:
                            # Client want to pull data from server, we need to upload
                            if self.step_before_upload():
                                self.step_after_upload(reign.upload())
                            else:
                                self.step_at_failed()
                        else:
                            self.step_at_unvalid_state()
                    # update regularly.
                    self.step_at_last()
                else:
                    del rg

            # left some time to maintainer lock
            time.sleep(openfed.SLEEP_SHORT_TIME)
        self.finish()
        return "Backend exited."

    @_backend_access
    def step_at_new_episode(self):
        pass

    @_backend_access
    def step_at_first(self):
        pass

    @_backend_access
    def step_at_zombine(self):
        pass

    @_backend_access
    def step_before_destroy(self) -> bool:
        return True

    @_backend_access
    def step_after_destroy(self, state=...):
        pass

    @_backend_access
    def step_before_download(self) -> bool:
        return True

    @_backend_access
    def step_after_download(self, state=...):

        assert self.aggregator is not None

        # fetch data from federeated core.
        packages = self.reign.tensor_indexed_packages
        task_info = self.reign.task_info

        # add received data to aggregator
        self.aggregator.step(packages, task_info)

        # increase the total received_numbers
        self.received_numbers += 1

        if openfed.VERBOSE.is_verbose:
            logger.info(f"Recieve Model\n"
                        f"@{self.received_numbers}\n"
                        f"From {self.reign}"
                        )

    @_backend_access
    def step_before_upload(self) -> bool:
        assert self.optimizer is not None
        assert self.aggregator is not None
        assert self.state_dict is not None

        # reset old data
        self.reign.reset()

        # pack new data
        self.reign.set_state_dict(self.state_dict)
        self.reign.pack_state(self.aggregator)
        self.reign.pack_state(self.optimizer)

        if openfed.DEBUG.is_debug:
            logger.info("Send")
        return True

    @_backend_access
    def step_after_upload(self, state=...):
        pass

    @_backend_access
    def step_at_last(self):
        """Related funtion to control the server state.
        """
        if self.received_numbers == 50:
            # the following code is just used for testing.
            # you should rewrite your logics code instead.
            task_info = self.aggregator.aggregate()
            self.aggregator.unpack_state(self.optimizer)
            self.optimizer.step()
            self.aggregator.zero_grad()
            self.received_numbers = 0
            self.last_aggregate_time = time.time()
            self.manual_stop()
        else:
            ...

    @_backend_access
    def step_at_unvalid_state(self):
        pass

    @_backend_access
    def step_at_failed(self):
        pass

    @_backend_access
    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Backend",
            description=str(self.maintainer)
        )
