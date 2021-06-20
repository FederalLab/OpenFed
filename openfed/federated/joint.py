import openfed

from ..common import Address, ConnectTimeout, SafeTread, logger
from ..utils import openfed_class_fmt
from .federated_world import FederatedWorld
from .register import register
from .reign import Reign
from .world import World


class Joint(SafeTread):
    """A thread to build connection among specified ends.
    """

    # Indicates whether the connection is established correctly.
    build_success: bool

    def __init__(self, address: Address, world: World):
        if address.rank == -1:
            if address.world_size == 2:
                address.rank = 1 if world.queen else 0
            else:
                msg = "Please specify the correct rank when world size is not 2"
                logger.error(msg)
                raise RuntimeError(msg)

        self.address = address
        self.build_success = False

        self.world = world

        SafeTread.__init__(self)
        # start this thread
        self.start()

        if self.world.queen:
            # if queen, wait until thread quit
            self.join()
            if not self.build_success:
                msg = f"Connect to {str(self.address)} failed."
                logger.error(msg)
                raise RuntimeError(msg)

    def safe_run(self):
        if openfed.VERBOSE.is_verbose:
            logger.info(f"Waiting for \n{repr(self.address)}")

        # create a federated world
        fed_world = FederatedWorld(self.world)

        # build the connection between the federated world
        try:
            fed_world.init_process_group(**self.address.as_dict)
        except ConnectTimeout as cte:
            del fed_world
            if openfed.DEBUG.is_debug:
                logger.error(str(cte))
            return f"Timeout {repr(self.address)}"

        # register the world
        with self.world.joint_lock:
            register.register_federated_world(fed_world, self.world)

        if self.address.world_size > 2:
            # rank is always set to 0 for that we want to build a
            # point2point connection between the master and each nodes.
            sub_pg_list = fed_world.build_point2point_group(rank=0)

            # bound pg with the federated world
            for sub_pg in sub_pg_list:
                reign = Reign(fed_world.get_store(
                    sub_pg), sub_pg, fed_world, self.world)
                with self.world.joint_lock:
                    self.world._pg_mapping[sub_pg] = reign
        else:
            # add the world group as reign if it is already a point to point connection.
            pg = fed_world._get_default_group()
            store = fed_world._get_default_store()
            reign = Reign(store, pg, fed_world, self.world)
            with self.world.joint_lock:
                self.world._pg_mapping[pg] = reign

        self.build_success = True

        if openfed.VERBOSE.is_verbose:
            logger.info(
                f"Connected\n{str(self.address)}")
        return f"Success! {repr(self.address)}"

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Joint",
            description=str(self.address),
        )

