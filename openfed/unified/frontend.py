import time
from typing import Dict, List, Union

from openfed.common import SLEEP_LONG_TIME, Address, default_address, logger
from openfed.federated import Maintainer, Reign, World
from openfed.utils import openfed_class_fmt
from torch import Tensor
from torch.optim import Optimizer

from .unify import Unify
from .utils import after_connection, before_connection, frontend_access


class Frontend(Unify):
    """An unified API of frontend for users.
    """
    @frontend_access
    def build_connection(self, world: World = None, address: Union[Address, List[Address]] = None, address_file: str = None):
        if world is None:
            world = World(leader=False)
        else:
            assert world.follower, "Frontend must be follower."

        if address is None:
            address = default_address

        self.world = world
        self.maintainer = Maintainer(self.world, address)
        while not Reign.default_reign():
            logger.debug("Wait for generating a valid reign")
            time.sleep(SLEEP_LONG_TIME.seconds)
        self.reign = Reign.default_reign()

        assert self.reign, "Reign not available!"

        # Auto register hooks for reign.
        # As for frontend, each frontend is only with 1 reign, which is specified here.
        # Thus, we can do the following thing once only.
        self._add_hook_to_reign()
        if self.state_dict is not None:
            self.reign.reset_state_dict(self.state_dict)

    @frontend_access
    @before_connection
    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        logger.debug(
            f"{'Set' if not self.state_dict else 'Unset'} state dict.")
        self.state_dict = state_dict

    @frontend_access
    @after_connection
    def pack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.pack_state(obj, keys)

    @frontend_access
    @after_connection
    def unpack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.unpack_state(obj, keys)

    @frontend_access
    @after_connection
    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Frontend",
            description=str(self.maintainer)
        )
