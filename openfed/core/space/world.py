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


import threading
from collections import OrderedDict
from enum import Enum, unique
from typing import Any, Dict

from openfed.common import Array
from openfed.common.base import peeper
from openfed.utils import openfed_class_fmt
from torch._C._distributed_c10d import ProcessGroup


@unique
class _ROLE(Enum):
    LEADER   = True
    FOLLOWER = False


peeper.world_list = list()

class World(Array):
    """Relation map between World, Country and Delivery:
        World: n master, varied roles
        ├── Country-a: singe master, n client
        │   └── Delivery-1: single master, single client.
        └── Country-b
            ├── Delivery-1
            └── Delivery-2
    """
    # If you want to exist current World, set it False
    ALIVE: bool

    # Your role in this World.
    # You can have different roles in different Worlds.
    ROLE: _ROLE

    _pg_mapping: Dict[ProcessGroup, Any]

    # Use them to track processes group.
    _NULL_GP   : Any = None
    _current_pg: ProcessGroup

    # avoid the conflict while joint many new Countries to current World at the some time
    joint_lock = threading.Lock()

    def __init__(self, leader: bool = False) -> None:
        """
        Args: 
            leader: if True, set the world as leader. Once the role is specified, you cannot change it again.
        """
        peeper.world_list.append(self)

        self.ALIVE       = True
        self.ROLE        = _ROLE.FOLLOWER if not leader else _ROLE.LEADER
        self._pg_mapping = OrderedDict()
        self._current_pg = self._NULL_GP

        super().__init__(self._pg_mapping, self.joint_lock)

    @classmethod
    def clear_world(cls) -> None:
        """Kill all world in _world_list with force.
        It is not safe to call this, but it can make you exit OpenFed env as soon as possible.
        """
        for world in peeper.world_list:
            world.killed()

    def killed(self) -> None:
        """Shout down this world with force. 
        If any delivery still uses, make them offline directly.
        """
        for _, delivery in self:
            delivery.offline()
        else:
            self.ALIVE = False

    @property
    def leader(self) -> bool:
        return self.ROLE == _ROLE.LEADER

    @property
    def follower(self) -> bool:
        return self.ROLE == _ROLE.FOLLOWER

    @property
    def default_delivery(self) -> Any:
        return self.default_value

    @property
    def default_pg(self) -> ProcessGroup:
        pg = self.default_key
        return pg if pg is not None else self._NULL_GP

    def is_valid_process_group(self, pg: ProcessGroup) -> bool:
        return pg is not self._NULL_GP and pg in self._pg_mapping

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name  = "World",
            description = (
                f"ROLE: {self.ROLE.value}\n"
                f"{len(self)} process groups are alive.\n"
            )
        )
