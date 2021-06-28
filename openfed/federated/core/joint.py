import time

from openfed.common import Address, SafeTread, logger
from openfed.utils import openfed_class_fmt

from ..reign import Reign
from ..space import Country, World
from ..utils.exception import BuildReignFailed, ConnectTimeout
from ..utils.register import register


class Joint(SafeTread):
    """A thread to build connection among specified ends.
    """

    # Indicates whether the connection is established correctly.
    build_success: bool

    def __init__(self, address: Address, world: World, auto_start: bool = True) -> None:
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
        if auto_start:
            self.start()

            if self.world.queen:
                # if queen, wait until thread quit
                self.join()
                if not self.build_success:
                    msg = f"Connect to {str(self.address)} failed."
                    logger.error(msg)
                    raise RuntimeError(msg)

    def safe_run(self) -> str:
        logger.info(f"Waiting\n{repr(self.address)}")

        # create a country
        country = Country(self.world)

        # build the connection between the country
        try:
            country.init_process_group(**self.address.as_dict)
        except ConnectTimeout as cte:
            del country
            logger.debug(cte)
            return f"Timeout {repr(self.address)}"

        # register the world
        with self.world.joint_lock:
            register.register_country(country, self.world)

        world_size = self.address.world_size
        if world_size > 2:
            if world_size > 10 and self.address.init_method.startswith("tcp"):
                msg = ("TCP Overload\n"
                       "There are too many node in tcp mode, which is not allowed.\n"
                       f"Make the world size smaller than 10 ({world_size} is given.) to run stablely.\n"
                       "Or use a share file system to initialize.\n"
                       "For example: ```--init_method file:///tmp/openfed.sharefile```.\n")
                logger.error(msg)
                raise RuntimeError(msg)

            # rank is always set to 0 for that we want to build a
            # point2point connection between the master and each nodes.
            sub_pg_list = country.build_point2point_group(rank=0)

            self.build_success = False
            # bound pg with the country
            for sub_pg in sub_pg_list:
                try:
                    reign = Reign(country.get_store(
                        sub_pg), sub_pg, country, self.world)
                    # it may failed to create connection sometimes between same subprocess.
                    # if any is success, we take it okay.
                    self.build_success = True
                except BuildReignFailed as e:
                    logger.exception(e)
                    continue
                with self.world.joint_lock:
                    self.world._pg_mapping[sub_pg] = reign

                # python(5766,0x70000fe24000) malloc: can't allocate region
                # :*** mach_vm_map(size=5639989190273028096, flags: 100) failed (error code=3)
                # python(5766,0x70000fe24000) malloc: *** set a breakpoint in malloc_error_break to debug
                # The following will make openfed more stable under tcp mode.
                time.sleep(0.1)
        else:
            # add the world group as reign if it is already a point to point connection.
            pg = country._get_default_group()
            store = country._get_default_store()
            try:
                reign = Reign(store, pg, country, self.world)
                with self.world.joint_lock:
                    self.world._pg_mapping[pg] = reign
                self.build_success = True
            except BuildReignFailed as e:
                logger.exception(e)
                self.build_success = False

        if self.build_success:
            logger.info(f"Connected\n{str(self.address)}")
        return f"Success! {repr(self.address)}" if self.build_success else f"Failed! {repr(self.address)}"

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Joint",
            description=str(self.address),
        )
