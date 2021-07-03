# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from openfed.common import logger
from openfed.utils import tablist

from ..reign import Reign
from ..space import ProcessGroup, World
from ..space.world import _world_list
from ..utils.register import register


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
        logger.debug(
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
            logger.debug(e)
            return False
