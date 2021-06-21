import time
from typing import Any, Dict, List, Union

from torch import Tensor
from torch.optim import Optimizer

import openfed
from openfed.common import Address, Peeper, default_address, logger
from openfed.common.constants import SLEEP_LONG_TIME
from openfed.common.unify import Unify, _frontend_access
from openfed.federated import Maintainer, Reign, World
from openfed.utils import openfed_class_fmt


class Frontend(Unify, Peeper):
    """An unified API of frontend for users.
    """
    maintainer: Maintainer

    reign: Reign
    frontend: bool = True

    @_frontend_access
    def build_connection(self, world: World = None, address: Union[Address, List[Address]] = None, address_file: str = None):
        if world is None:
            world = World(king=False)
        else:
            assert world.queen, "Frontend must be queen."

        if address is None:
            address = default_address

        self.world = world
        self.maintainer = Maintainer(self.world, address)
        while not Reign.default_reign():
            if openfed.VERBOSE.is_verbose:
                logger.info("Wait for generating a valid reign")
            time.sleep(SLEEP_LONG_TIME)
        self.reign = Reign.default_reign()

    @_frontend_access
    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        self.reign.set_state_dict(state_dict)

    @_frontend_access
    def pack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.pack_state(obj, keys)

    @_frontend_access
    def unpack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.unpack_state(obj, keys)

    @_frontend_access
    def upload(self) -> bool:
        state = self.reign.upload()
        if not state:
            msg = "Upload Failed."
            logger.info(msg)
        return state

    @_frontend_access
    def download(self) -> bool:
        state = self.reign.download()
        if not state:
            msg = "Download Falied."
            logger.info(msg)
        return state

    @_frontend_access
    def set_task_info(self, task_info: Dict) -> None:
        self.reign.set_task_info(task_info)

    @_frontend_access
    def get_task_info(self) -> Dict:
        return self.reign.task_info

    @_frontend_access
    def set(self, key: str, value: Any) -> None:
        self.reign.set(key, value)

    @_frontend_access
    def get(self, key: str) -> Any:
        return self.reign.get(key)

    @_frontend_access
    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Frontend",
            description=str(self.maintainer)
        )
