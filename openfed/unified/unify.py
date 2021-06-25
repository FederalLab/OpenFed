from typing import Dict

import openfed
from loguru import logger
from openfed.common.thread import SafeTread
from openfed.federated.destroy import Destroy
from openfed.federated.maintainer import Maintainer
from openfed.federated.reign import Reign
from openfed.federated.world import World
from openfed.unified.utils import _backend_access
from openfed.utils import openfed_class_fmt
from openfed.utils.keyboard import keyboard_interrupt_handle
from torch import Tensor


class Unify(object):
    """Provide a unified api for backend and frontend.
    """
    maintainer: Maintainer

    reign: Reign

    world: World

    version: int

    frontend: bool
    # fontend xor backward == True
    backend: bool

    async_op: bool

    dynamic_address_loading: bool

    def __init__(self,
                 frontend: bool = True,
                 capture_keyboard_interrupt: bool = True,
                 async_op_beckend: bool = True,
                 dynamic_address_loading: bool = True,
                 register_default_step_for_backend: bool = True):
        """Whether act as a frontend.
        Frontend is always in sync mode, which will ease the coding burden.
        Backend will be set as async mode by default.
        """
        self.frontend = frontend
        # Set a flag for backend.
        self.backend = not self.frontend
        if capture_keyboard_interrupt:
            logger.debug("OpenFed will capture keyboard interrupt signal.")
            keyboard_interrupt_handle()

        if self.frontend:
            self.async_op = False
        elif async_op_beckend:
            self.async_op = True
        else:
            self.async_op = False

        self.dynamic_address_loading = dynamic_address_loading
        self.register_default_step_for_backend = register_default_step_for_backend

        # Set default value
        self.version = 0

    @property
    def nick_name(self) -> str:
        return self.reign.nick_name

    def build_connection(self, *args, **kwargs):
        raise NotImplementedError

    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        raise NotImplementedError

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Unify",
            description="The unify api for backend and frontend."
        )

    def finish(self):
        Destroy.destroy_all_in_a_world(self.world)
        if self.maintainer:
            self.maintainer.manual_stop()

        if not self.frontend:
            logger.info(f"Finished.\n {self}")
            exit(0)

    @_backend_access
    def run(self, *args, **kwargs):
        return SafeTread.run(self, *args, **kwargs)

    def __enter__(self):
        self.old_variable_list = [
            openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading,
            openfed.ASYNC_OP.is_async_op,
        ]

        openfed.DYNAMIC_ADDRESS_LOADING.set(self.dynamic_address_loading)
        openfed.ASYNC_OP.set(self.async_op)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore state
        dynamic_address_loading, async_op = self.old_variable_list

        openfed.DYNAMIC_ADDRESS_LOADING.set(dynamic_address_loading)
        openfed.ASYNC_OP.set(async_op)
