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


import time

from openfed.common import Address_, SafeThread, logger
from openfed.utils import openfed_class_fmt

from ..delivery import Delivery
from ..space import Country, World
from ..utils.exceptions import BuilddeliveryFailed, ConnectTimeout
from ..utils.register import register


class Joint(SafeThread):
    """A thread to build connection among specified ends.
    """

    # Indicates whether the connection is established correctly.
    build_success: bool

    def __init__(self, address: Address_, world: World, auto_start: bool = True) -> None:
        if address.rank == -1:
            if address.world_size == 2:
                address.rank = 1 if world.follower else 0  # type: ignore
            else:
                msg = "Please specify the correct rank when world size is not 2"
                logger.error(msg)
                raise RuntimeError(msg)

        self.address = address
        self.build_success = False
        self.world = world

        SafeThread.__init__(self)
        # start this thread
        if auto_start:
            self.start()

            if self.world.follower:
                # if follower, wait until thread quit
                self.join()
                if not self.build_success:
                    raise RuntimeError(f"Connect to {self.address} failed.")

    def safe_run(self) -> str:
        logger.debug(f"Waiting ...")

        # create a country
        country = Country(self.world)

        # build the connection between the country
        try:
            country.init_process_group(**self.address.address)
        except ConnectTimeout as cte:
            del country
            logger.debug(cte)
            return f"Timeout {self.address}"

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
                raise RuntimeError(msg)

            # rank is always set to 0 for that we want to build a
            # point2point connection between the master and each nodes.
            sub_pg_list = country.build_point2point_group(rank=0)

            self.build_success = False
            # bound pg with the country
            for sub_pg in sub_pg_list:
                try:
                    delivery = Delivery(country.get_store(
                        sub_pg), sub_pg, country, self.world)
                    # it may failed to create connection sometimes between same subprocess.
                    # if any is success, we take it okay.
                    self.build_success = True
                except BuilddeliveryFailed as e:
                    logger.debug(e)
                    continue
                with self.world.joint_lock:
                    self.world._pg_mapping[sub_pg] = delivery

                # python(5766,0x70000fe24000) malloc: can't allocate region
                # :*** mach_vm_map(size=5639989190273028096, flags: 100) failed (error code=3)
                # python(5766,0x70000fe24000) malloc: *** set a breakpoint in malloc_error_break to debug
                # The following will make openfed more stable under tcp mode.
                time.sleep(0.1)
        else:
            # add the world group as delivery if it is already a point to point connection.
            pg = country._get_default_group()
            store = country._get_default_store()
            try:
                delivery = Delivery(store, pg, country, self.world)
                with self.world.joint_lock:
                    self.world._pg_mapping[pg] = delivery
                self.build_success = True
            except BuilddeliveryFailed as e:
                logger.debug(e)
                self.build_success = False

        if self.build_success:
            logger.debug(f"Connected\n{str(self.address)}")
        return f"Success! {self.address}" if self.build_success else f"Failed! {self.address}"

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Joint",
            description=self.address,
        )
