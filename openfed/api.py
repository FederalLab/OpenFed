# Provide a unified api for users on backend and frontend.
from typing import Dict, List, Union

from torch import Tensor

from openfed.common.address import Address
from openfed.federated.world import World
from openfed.unified.backend import Backend
from openfed.unified.frontend import Frontend
from openfed.utils import openfed_class_fmt


class API(Frontend, Backend):
    """Provide a unified API for users on backend and frontend.
    """

    def build_connection(self,
                         world: World = None,
                         address: Union[Address, List[Address]] = None,
                         address_file: str = None):
        if self.frontend:
            return Frontend.build_connection(self, world, address, address_file)
        else:
            return Backend.build_connection(self, world, address, address_file)

    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        """net.state_dict(keep_vars=True), keep_vars=True is very importance.
        """
        if self.frontend:
            return Frontend.set_state_dict(self, state_dict)
        else:
            return Backend.set_state_dict(self, state_dict)

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="OpenFed Unified API",
            description=str(self.maintainer)
        )
