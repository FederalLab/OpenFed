import openfed

from ..common import logger
from .federated_world import ProcessGroup
from .register import register
from .reign import Reign
from .world import World


class Destroy(object):
    """Automatically destroy a process group along with its world.
    """
    @classmethod
    def destroy(cls, pg: ProcessGroup, world: World = None):
        if world is None:
            world = register.default_world

        if pg == world._current_pg:
            world._current_pg = world._NULL_GP

        reign = world._pg_mapping[pg]
        reign.offline()
        del world._pg_mapping[pg]

        federated_world = reign.federated_world
        federated_world.destroy_process_group(pg)

        if not federated_world.is_initialized() or federated_world._group_count == 1:
            if openfed.DEBUG.is_debug:
                logger.info(f"Destroy {federated_world}")
            register.deleted_federated_world(federated_world)
        else:
            ...

    @classmethod
    def destroy_current(cls, world: World = None):
        if world is None:
            world = register.default_world
        cls.destroy(world._current_pg, world)

    @classmethod
    def destroy_all_in_a_world(cls, world: World = None):
        if world is None:
            world = register.default_world
        for pg, _ in world:
            if pg is not None:
                cls.destroy(pg, world)

    @classmethod
    def destroy_all_in_all_world(cls):
        """A safe way to destroy all federated world which has been registered.
        """
        for _, world in register:
            if world is not None:
                cls.destroy_all_in_a_world(world)

    @classmethod
    def destroy_reign(cls, reign: Reign):
        cls.destroy(reign.pg, reign.world)
        del reign
