import openfed
import openfed.common.logging as logger
from openfed.federated.country import ProcessGroup
from openfed.federated.register import register
from openfed.federated.reign import Reign
from openfed.federated.world import World, _world_list
from openfed.utils.table import tablist


class Destroy(object):
    """Automatically destroy a process group along with its world.
    """
    @classmethod
    def destroy(cls, pg: ProcessGroup, world: World = None) -> None:
        if world is None:
            world = register.default_world

        if pg == world._current_pg:
            world._current_pg = world._NULL_GP

        reign = world._pg_mapping[pg]
        reign.offline()
        del world._pg_mapping[pg]

        country = reign.country
        country.destroy_process_group(pg)

        if not country.is_initialized() or country._group_count == 1:
            logger.debug(f"Destroy {country}")
            register.deleted_country(country)
        else:
            ...

    @classmethod
    def destroy_current(cls, world: World = None) -> None:
        if world is None:
            world = register.default_world
        cls.destroy(world._current_pg, world)

    @classmethod
    def destroy_all_in_a_world(cls, world: World = None) -> None:
        if world is None:
            world = register.default_world
        for pg, _ in world:
            if pg is not None:
                cls.destroy(pg, world)

    @classmethod
    def destroy_all_in_all_world(cls) -> None:
        """A safe way to destroy all country which has been registered.
        """
        logger.warning(
            "Destroy OpenFed\n" +
            tablist(
                head=["World", "Country", "Reign"],
                data=[len(_world_list),
                      len(register),
                      sum([len(w) for w in _world_list])]
            )
        )
        for _, world in register:
            if world is not None:
                cls.destroy_all_in_a_world(world)

    @classmethod
    def destroy_reign(cls, reign: Reign) -> bool:
        try:
            cls.destroy(reign.pg, reign.world)
            del reign
            return True
        except Exception as e:
            logger.exception(e)
            return False
