import time
from typing import Any, Dict, List, Union, overload

from torch import Tensor
from torch.optim import Optimizer

import openfed
from openfed.common import Address, Peeper, default_address, logger
from openfed.common.constants import SLEEP_LONG_TIME
from openfed.federated import Destroy, Maintainer, Reign, World
from openfed.utils import openfed_class_fmt


class Frontend(Peeper):
    """An unified API for users.
    """
    maintainer: Maintainer

    reign: Reign

    @overload
    def __init__(self):
        """
            Build a default frontend for fast testing.
        """

    @overload
    def __init__(self,
                 world: World,
                 address: Address):
        """
            Build a frontend with world and address.
        """

    def __init__(self, **kwargs):
        world = kwargs.get('world', None)
        address = kwargs.get('address', None)
        if world is None:
            world = World(king=False)
        else:
            world = kwargs['world']
            assert world.queen, "Frontend must be queen."

        if address is None:
            address = default_address

        self.world = world
        self.build_connection(address)

    def build_connection(self, address: Address):
        self.maintainer = Maintainer(self.world, address)
        while not Reign.default_reign():
            if openfed.VERBOSE.is_verbose:
                logger.info("Wait for generating a valid reign")
            time.sleep(SLEEP_LONG_TIME)
        self.reign = Reign.default_reign()

    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        self.reign.set_state_dict(state_dict)

    def pack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.pack_state(obj, keys)

    def unpack_state(self, obj: Optimizer, keys: Union[str, List[str]] = None):
        self.reign.unpack_state(obj, keys)

    def upload(self) -> bool:
        state = self.reign.upload()
        if not state:
            msg = "Upload Failed."
            logger.info(msg)
        return state

    def download(self) -> bool:
        state = self.reign.download()
        if not state:
            msg = "Download Falied."
            logger.info(msg)
        return state

    def set_task_info(self, task_info: Dict) -> None:
        self.reign.set_task_info(task_info)

    def get_task_info(self) -> Dict:
        return self.reign.task_info

    def set(self, key: str, value: Any) -> None:
        self.reign.set(key, value)

    def get(self, key: str) -> Any:
        return self.reign.get(key)

    def finish(self):
        if self.reign is not None:
            Destroy.destroy_reign(self.reign)

        self.maintainer.manual_stop()

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Frontend",
            description=str(self.maintainer)
        )
