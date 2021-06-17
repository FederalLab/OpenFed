from collections import OrderedDict
from typing import Dict
import openfed
# from .core.federated_c10d import FederatedWorld
from .world import World

# 循环import的问题
FederatedWorld = object
# At most case, you are not allowed to modifed this list manually.
# FederatedWorld是底层的通讯抽象，World是对应的参数配置
__federated_world__: Dict[FederatedWorld, World] = OrderedDict()


class _Register(object):
    @classmethod
    def register_federated_world(cls, federated_world: FederatedWorld, world: World):
        if federated_world in __federated_world__:
            raise KeyError("Already registered.")
        else:
            __federated_world__[federated_world] = world

    @classmethod
    def deleted_federated_world(cls, federated_world: FederatedWorld):
        if federated_world in __federated_world__:
            if federated_world.is_initialized():
                if openfed.VERBOSE:
                    print("Try to destroy all process group in federated world.")
                federated_world.destroy_process_group(
                    group=federated_world.WORLD)
            del __federated_world__[federated_world]

    @classmethod
    def deleted_all_federated_world(cls):
        if openfed.VERBOSE:
            print("Try to delete all process group in all federated world.")
        for k in __federated_world__:
            cls.deleted_federated_world(k)

    @classmethod
    def is_registered(cls, federated_world: FederatedWorld) -> bool:
        return federated_world in __federated_world__

    def __iter__(self):
        return zip(__federated_world__.keys(), __federated_world__.values())

    @property
    def default_federated_world(cls) -> FederatedWorld:
        """ If not exists, return None
        """
        for fed_world in __federated_world__:
            return fed_world
        else:
            return None

    @property
    def default_world(cls) -> World:
        """If not exists, return None
        """
        for fed_world in __federated_world__:
            return __federated_world__[fed_world]
        else:
            return None

    def __len__(self):
        return len(__federated_world__)


register = _Register()
