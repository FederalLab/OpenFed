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
from typing import Any, Dict, List, TypeVar

from openfed.common import Array
from openfed.utils import openfed_class_fmt
from torch._C._distributed_c10d import ProcessGroup


@unique
class _ROLE(Enum):
    LEADER = True
    FOLLOWER = False


_W = TypeVar("_W", bound="World")
_world_list: List[_W] = []


class Reign():
    """Define here for better code analysis. 
    Refer openfed.federated.federated.Reign for more details about this class.
    """
    ...


class World(Array):
    """Relation map between World, Country and Reign:
        World: n master, varied roles
        ├── Country-a: singe master, n client
        │   └── Reign-1: single master, single client.
        └── Country-b
            ├── Reign-1
            └── Reign-2
    """
    # If you want to exist current World, set it False
    ALIVE: bool

    # Your role in this World.
    # You can have different roles in different Worlds.
    ROLE: _ROLE

    _pg_mapping: Dict[ProcessGroup, Reign]

    # Use them to track processes group.
    _NULL_GP: Any = None
    _current_pg: ProcessGroup

    # avoid the conflict while joint many new Countries to current World at the some time
    joint_lock = threading.Lock()

    def __init__(self, leader: bool = False) -> None:
        """
        Args: 
            leader: if True, set the world as leader. Once the role is specified, you cannot change it again.
        """
        _world_list.append(self)

        self.ALIVE = True
        self.ROLE = _ROLE.FOLLOWER if not leader else _ROLE.LEADER
        self._pg_mapping = OrderedDict()
        self._current_pg = self._NULL_GP

        super().__init__(self._pg_mapping, self.joint_lock)

    @classmethod
    def clear_world(cls) -> None:
        """Kill all world in _world_list with force.
        It is not safe to call this, but it can make you exit OpenFed env as soon as possible.
        """
        for world in _world_list:
            world.killed()

    def killed(self) -> None:
        """Shout down this world with force. 
        If any reign still uses, make them offline directly.
        """
        for _, reign in self:
            reign.offline()
        else:
            self.ALIVE = False

    @property
    def leader(self) -> bool:
        return self.ROLE == _ROLE.LEADER

    @property
    def follower(self) -> bool:
        return self.ROLE == _ROLE.FOLLOWER

    @property
    def default_reign(self) -> Reign:
        return self.default_values

    @property
    def default_pg(self) -> ProcessGroup:
        pg = self.default_keys
        return pg if pg is not None else self._NULL_GP

    def is_valid_process_group(self, pg: ProcessGroup) -> bool:
        return pg is not self._NULL_GP and pg in self._pg_mapping

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="World",
            description=(
                f"ROLE: {self.ROLE.value}\n"
                f"{len(self)} Process Group Alive.\n"
            )
        )
