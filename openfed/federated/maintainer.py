import time
from threading import Lock
from typing import Dict, List, Union

import openfed
from openfed.common import Address, Array, SafeTread, logger
from openfed.federated.joint import Joint
from openfed.federated.lock import add_maintainer_lock, del_maintainer_lock
from openfed.federated.world import World
from openfed.utils import openfed_class_fmt


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
    # The shared information among all country in this maintainer.
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
                        f"Error Address"
                        f"{str(add)}"
                        f"Discarded.")
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
                                        "Error Address\n"
                                        f"{str(address)}"
                                        f"Discarded.")
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

    def clear_finished_queue(self):
        self.finished_queue.clear()

    def clear_discard_queue(self):
        self.discard_queue.clear()
