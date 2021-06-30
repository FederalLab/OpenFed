import time
from typing import Any, Dict, List, Union

import openfed
from openfed.common import SLEEP_LONG_TIME, Address, default_address, logger
from openfed.federated import Maintainer, Reign, World
from openfed.utils import openfed_class_fmt
from torch import Tensor
from torch.optim import Optimizer

from .unify import Unify
from .utils import frontend_access, after_connection, before_connection


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
    def _wait_handler(self, flag: bool):
        if flag:
            return True
        elif openfed.ASYNC_OP.is_async_op:
            while not self.reign.deal_with_hang_up():
                if self.reign.is_offline:
                    return False
                time.sleep(openfed.SLEEP_SHORT_TIME.seconds)
            return True

    @frontend_access
    @after_connection
    def upload(self) -> bool:
        """As for frontend, it is much easier for us to judge the new version.
        A download and upload is build a version updating.
        So increase version number here.
        """
        return self._wait_handler(self.reign.upload(self.version))

    @frontend_access
    @after_connection
    def update_version(self, version: int = None):
        """Update inner model version.
        """
        if version:
            self.version = version
        else:
            self.version += 1

    @frontend_access
    @after_connection
    def download(self) -> bool:
        return self._wait_handler(self.reign.download(self.version))

    @frontend_access
    @after_connection
    def set_task_info(self, task_info: Dict) -> None:
        self.reign.set_task_info(task_info)

    @frontend_access
    @after_connection
    def get_task_info(self) -> Dict:
        return self.reign.task_info

    @frontend_access
    @after_connection
    def set(self, key: str, value: Any) -> None:
        self.reign.set(key, value)

    @frontend_access
    @after_connection
    def get(self, key: str) -> Any:
        return self.reign.get(key)

    @frontend_access
    @after_connection
    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Frontend",
            description=str(self.maintainer)
        )
