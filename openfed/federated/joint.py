import openfed
from openfed.common import Address, ConnectTimeout, SafeTread, logger
from openfed.federated.country import Country
from openfed.federated.register import register
from openfed.federated.reign import Reign
from openfed.federated.world import World
from openfed.utils import openfed_class_fmt


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
            logger.info(f"Waiting\n{repr(self.address)}")

        # create a country
        country = Country(self.world)

        # build the connection between the country
        try:
            country.init_process_group(**self.address.as_dict)
        except ConnectTimeout as cte:
            del country
            if openfed.DEBUG.is_debug:
                logger.error(str(cte))
            return f"Timeout {repr(self.address)}"

        # register the world
        with self.world.joint_lock:
            register.register_country(country, self.world)

        if self.address.world_size > 2:
            # rank is always set to 0 for that we want to build a
            # point2point connection between the master and each nodes.
            sub_pg_list = country.build_point2point_group(rank=0)

            # bound pg with the country
            for sub_pg in sub_pg_list:
                reign = Reign(country.get_store(
                    sub_pg), sub_pg, country, self.world)
                with self.world.joint_lock:
                    self.world._pg_mapping[sub_pg] = reign
        else:
            # add the world group as reign if it is already a point to point connection.
            pg = country._get_default_group()
            store = country._get_default_store()
            reign = Reign(store, pg, country, self.world)
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
