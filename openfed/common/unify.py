from typing import Dict

from openfed.common.logging import logger
from openfed.common.thread import SafeTread
from openfed.common.vars import DEBUG, VERBOSE
from openfed.federated.destroy import Destroy
from openfed.federated.maintainer import Maintainer
from openfed.federated.reign import Reign
from openfed.federated.world import World
from openfed.utils import openfed_class_fmt
from openfed.utils.keyboard_interrupt_handle import keyboard_interrupt_handle
from torch import Tensor


def _frontend_access(func):
    def wrapper(self, *args, **kwargs):
        if not self.frontend:
            if DEBUG.is_debug:
                logger.warning("frontend funciton.")
            return None
        else:
            return func(self, *args, **kwargs)
    return wrapper


def _backend_access(func):
    def wrapper(self, *args, **kwargs):
        if self.frontend:
            if DEBUG.is_debug:
                logger.warning("backend funciton.")
            return None
        else:
            return func(self, *args, **kwargs)
    return wrapper


class Unify(object):
    """Provide a unified api for backend and frontend.
    """
    maintainer: Maintainer

    reign: Reign

    world: World

    version: int

    frontend: bool

    def __init__(self,
                 frontend: bool = True,
                 capture_keyboard_interrupt: bool = True):
        """Whether act as a frontend.
        """
        self.frontend = frontend
        if capture_keyboard_interrupt:
            if DEBUG.is_debug:
                logger.info("OpenFed will capture keyboard interrupt signal.")
            keyboard_interrupt_handle()

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
        self.maintainer.manual_stop()

        if not self.frontend:
            if VERBOSE.is_verbose:
                logger.info("Finished.\n"+str(self))
            exit(0)

    @_backend_access
    def run(self, *args, **kwargs):
        return SafeTread.run(self, *args, **kwargs)
