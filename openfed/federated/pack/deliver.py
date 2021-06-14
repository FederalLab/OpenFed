from openfed.utils.types import PACKAGES
from openfed.federated.core.federated_c10d import ProcessGroup, FederatedWorld
from openfed.federated.register import World
from openfed.federated.core.functional import gather_object


class Delivery(object):
    """负责数据交互
    """
    pg: ProcessGroup
    federated_world: FederatedWorld
    world: World

    def __init__(self, pg: ProcessGroup, federated_world: FederatedWorld, world: World):
        self.pg = pg
        self.federated_world = federated_world
        self.world = world

    def pull(self) -> PACKAGES:
        """从另一端拉取数据。
        """
        assert self.federated_world._get_group_size(
            self.pg) == 2, "Delivery is designed for group with size 2"

        received = [None, None]
        rank = 0 if self.world.is_king() else 1

        gather_object(None, received, dst=rank, group=self.pg,
                      federated_world=self.federated_world)

        return received[rank]

    def push(self, packages: PACKAGES) -> None:
        """向另一段发送数据。
        """
        assert self.federated_world._get_group_size(
            self.pg) == 2, "Delivery is designed for group with size 2"

        rank = 1 if self.world.is_king() else 0
        gather_object(packages, [None, None], dst=rank,
                      group=self.pg, federated_world=self.federated)
