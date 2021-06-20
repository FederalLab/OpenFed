import time
from threading import Lock
from typing import Dict, List, Union

import openfed

from ..common import Address, Array, ConnectTimeout, SafeTread, logger
from ..utils import openfed_class_fmt
from .federated_world import FederatedWorld, ProcessGroup
from .lock import add_maintainer_lock, del_maintainer_lock
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
            description=repr(self.address),
        )


class Maintainer(Array, SafeTread):
    """
    Dynamic build the connection.
    """
    # unfinished address
    # Address -> [last try time, try_cnt]
    pending_queue: Dict[Address, List[Union[float, int]]]

    # finished address
    # Address -> [build time, try_cnt]
    finished_queue: Dict[Address, List[Union[float, int]]]

    # discard address
    # Address -> [last try time, try_cnt]
    discard_queue: Dict[Address, List[Union[float, int]]]

    maintainer_lock: Lock
    # The shared information among all federated world in this maintainer.
    world: World

    def __init__(self,
                 world: World,
                 address: Union[Address, List[Address]] = None,
                 address_file: str = None):
        """
            Only a single valid address is allowed in client.
        """
        self.pending_queue = dict()
        self.finished_queue = dict()
        self.discard_queue = dict()

        Array.__init__(self, self.pending_queue)

        self.maintainer_lock = Lock()
        add_maintainer_lock(self, self.maintainer_lock)

        self.world = world

        self.address_file = address_file

        if address is not None:
            if not isinstance(address, (list, tuple)):
                address = [address]
        else:
            address = []

        for add in address:
            self.pending_queue[add] = [time.time(), 0]

        self.read_address_from_file()
        # call here
        SafeTread.__init__(self)

        if self.world.king:
            self.start()
            if not openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                self.join()
        else:
            if len(self) > 1:
                msg = "Too many fed addr are specified. Only allowed 1."
                logger.error(msg)
                raise RuntimeError(msg)
            elif len(self) == 1:
                address, (last_time, try_cnt) = self[0]
                Joint(address, self.world)
                del self.pending_queue[address]
                self.finished_queue[address] = [time.time(), try_cnt+1]
            else:
                if openfed.VERBOSE.is_verbose:
                    logger.info("Waiting for a valid address")

    def read_address_from_file(self) -> None:
        if self.address_file is None:
            return

        address_list = Address.read_address_from_file(self.address_file)

        for add in address_list:
            if add in self.pending_queue:
                # already in pending queue
                ...
            elif add in self.finished_queue:
                # already connected
                ...
            elif add in self.discard_queue:
                if openfed.DEBUG.is_debug:
                    logger.error(
                        f"{repr(add)} is not a valid address, please remove it from your address list.")
            else:
                # add address to pending queue
                self.pending_queue[add] = [time.time(), 0]

    def safe_run(self):
        while not self.stopped and self.world.ALIVE:
            # update pending list
            self.read_address_from_file()

            def try_now(last_time, try_cnt) -> bool:
                if time.time() - last_time < openfed.INTERVAL_AFTER_LAST_FAILED_TIME:
                    return False
                if try_cnt > openfed.MAX_TRY_TIMES:
                    return False
                return True

            if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                for address,  (last_time, try_cnt) in self:
                    if try_now(last_time, try_cnt):
                        joint = Joint(address, self.world)
                        joint.join()
                        if joint.build_success:
                            self.finished_queue[address] = [
                                time.time(), try_cnt + 1]
                            del self.pending_queue[address]
                        else:
                            try_cnt += 1
                            if try_cnt > openfed.MAX_TRY_TIMES:
                                # move to discard queue
                                if openfed.VERBOSE.is_verbose:
                                    logger.error(
                                        f"Error address {repr(address)}, please remove it from your address list")
                                self.discard_queue[address] = [
                                    time.time(), try_cnt]
                                del self.pending_queue[address]
                            else:
                                self.pending_queue[address] = [
                                    time.time(), try_cnt]
            else:
                joint_address_mapping = []
                for address in self.pending_queue:
                    joint = Joint(address, self.world)
                    joint_address_mapping.append([joint, address])

                for joint, address in joint_address_mapping:
                    joint.join()
                    if joint.build_success:
                        self.finished_queue[address] = [time.time(), 1]
                        del self.pending_queue[address]
                    else:
                        self.discard_queue[address] = [time.time(), 1]
                        del self.pending_queue[address]
                        msg = f"Failed build connection with {address}"
                        if openfed.DEBUG.is_debug:
                            raise RuntimeError(msg)
                        else:
                            logger.error(msg)

            if len(self) == 0:
                if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                    time.sleep(openfed.SLEEP_LONG_TIME)
                else:
                    return f"Success: {len(self.finished_queue)} new federeated world added."
            else:
                if not openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                    raise RuntimeError(f"Failed: {len(self.discard_queue)}")
                else:
                    time.sleep(openfed.SLEEP_LONG_TIME)
        return "Force Quit XXX" + str(self)

    def kill_world(self):
        self.world.killed()

    def manual_stop(self, kill_world: bool = True):
        if kill_world:
            self.kill_world()
        super().manual_stop()

    def manual_joint(self, address: Address):
        if not openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading and self.world.king:
            raise RuntimeError("Dynamic loading is not allowed!")

        if openfed.VERBOSE.is_verbose:
            logger.info(f"Manually add a new address: {repr(address)}")
        if self.world.king:
            self.pending_queue[address] = [time.time(), 0]
        else:
            Joint(address, self.world)

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Maintainer",
            description=(
                f"{len(self.pending_queue)} in pending\n"
                f"{len(self.finished_queue)} in finished\n"
                f"{len(self.discard_queue)} in discard\n"
            )
        )

    def __del__(self):
        del_maintainer_lock(self)
        super().__del__()


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
