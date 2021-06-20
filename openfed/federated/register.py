from collections import OrderedDict
from typing import Dict

from ..common import Array, logger
from ..common.vars import DEBUG
from ..utils import openfed_class_fmt


class FederatedWorld():
    ...


class World():
    ...


# At most case, you are not allowed to modifed this list manually.
_federated_world: Dict[FederatedWorld, World] = OrderedDict()


class _Register(Array):

    def __init__(self):
        super(_Register, self).__init__(_federated_world)

    @classmethod
    def register_federated_world(cls, federated_world: FederatedWorld, world: World):
        if federated_world in _federated_world:
            raise KeyError("Already registered.")
        else:
            _federated_world[federated_world] = world

    @classmethod
    def deleted_federated_world(cls, federated_world: FederatedWorld):
        if federated_world in _federated_world:
            if federated_world.is_initialized():
                if DEBUG.is_debug:
                    logger.info(
                        f"Forece to delete federated world: {federated_world}")
                federated_world.destroy_process_group(
                    group=federated_world.WORLD)

            del _federated_world[federated_world]
            del federated_world

    @classmethod
    def is_registered(cls, federated_world: FederatedWorld) -> bool:
        return federated_world in _federated_world

    @property
    def default_federated_world(self) -> FederatedWorld:
        """ If not exists, return None
        """
        return self.default_keys

    @property
    def default_world(self) -> World:
        """If not exists, return None
        """
        return self.default_values

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Reigster",
            description=f"{len(self)} Federated World have been registed."
        )


register = _Register()
