import time
from threading import Lock
from typing import Dict, List, Union

import openfed
from openfed.common.address import Address
from openfed.common.array import Array
from openfed.common.logging import logger
from openfed.common.thread import SafeTread
from openfed.federated.joint import Joint
from openfed.federated.lock import add_maintainer_lock, del_maintainer_lock
from openfed.federated.world import World
from openfed.utils import openfed_class_fmt
from openfed.utils.table import tablist


class Maintainer(Array, SafeTread):
    """
    Dynamic build the connection.
    """
    # unfinished address
    # Address -> [last try time, try_cnt]
    pending_queue: Dict[str, List[Union[float, int, Address]]]

    # finished address
    # Address -> [build time, try_cnt]
    finished_queue: Dict[str, List[Union[float, int, Address]]]

    # discard address
    # Address -> [last try time, try_cnt]
    discard_queue: Dict[str, List[Union[float, int, Address]]]

    maintainer_lock: Lock
    # The shared information among all country in this maintainer.
    world: World

    abnormal_exited: bool

    def __init__(self,
                 world: World,
                 address: Union[Address, List[Address]] = None,
                 address_file: str = None) -> None:
        """
            Only a single valid address is allowed in client.
        """
        self.pending_queue = dict()
        self.finished_queue = dict()
        self.discard_queue = dict()
        self.abnormal_exited = False

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
            self.pending_queue[str(add)] = [time.time(), 0, add]

        self.read_address_from_file()
        # call here
        SafeTread.__init__(self)

        if self.world.king:
            self.start()
            if not openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                self.join()
                if self.abnormal_exited:
                    # raise error here, but not in self.safe_run()
                    raise RuntimeError(
                        "Errors occured while building connection to new address.")
        else:
            if len(self) > 1:
                msg = "Too many fed addr are specified. Only allowed 1."
                logger.error(msg)
                raise RuntimeError(msg)
            elif len(self) == 1:
                str_add, (last_time, try_cnt, address) = self[0]
                Joint(address, self.world)
                del self.pending_queue[str_add]
                self.finished_queue[str_add] = [
                    time.time(), try_cnt+1, address]
            else:
                if openfed.VERBOSE.is_verbose:
                    logger.info("Waiting for a valid address")

    def read_address_from_file(self) -> None:
        if self.address_file is None:
            return

        address_list = Address.load_from_file(self.address_file)

        for add in address_list:
            if str(add) in self.pending_queue:
                # already in pending queue
                ...
            elif str(add) in self.finished_queue:
                # already connected
                ...
            elif str(add) in self.discard_queue:
                if openfed.DEBUG.is_debug:
                    logger.error(
                        f"Error Address"
                        f"{str(add)}"
                        f"Discarded.")
            else:
                # add address to pending queue
                self.pending_queue[str(add)] = [time.time(), 0, add]

    def safe_run(self) -> str:
        while not self.stopped and self.world.ALIVE:
            # update pending list
            self.read_address_from_file()

            def try_now(last_time, try_cnt) -> bool:
                if time.time() - last_time < openfed.INTERVAL_AFTER_LAST_FAILED_TIME:
                    return False
                if try_cnt > openfed.MAX_TRY_TIMES:
                    return False
                return True

            for str_add, (last_time, try_cnt, address) in self:
                if try_now(last_time, try_cnt):
                    joint = Joint(address, self.world)
                    joint.join()
                    if joint.build_success:
                        self.finished_queue[str_add] = [
                            time.time(), try_cnt + 1, address]
                        del self.pending_queue[str_add]
                    else:
                        try_cnt += 1
                        if try_cnt > openfed.MAX_TRY_TIMES:
                            # move to discard queue
                            if openfed.VERBOSE.is_verbose:
                                logger.error(
                                    "Error Address\n"
                                    f"{str_add}"
                                    f"Discarded.")
                            self.discard_queue[str_add] = [
                                time.time(), try_cnt, address]
                            del self.pending_queue[str_add]
                            break
                        else:
                            self.pending_queue[str_add] = [
                                time.time(), try_cnt, address]

            if len(self) == 0:
                if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                    time.sleep(openfed.SLEEP_LONG_TIME)
                else:
                    return f"Success: {len(self.finished_queue)} new federeated world added."
            else:
                if not openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                    if len(self.discard_queue) != 0:
                        self.abnormal_exited = True
                        break
                else:
                    time.sleep(openfed.SLEEP_LONG_TIME)
            if openfed.DEBUG.is_debug:
                logger.info(str(self))
        return "Force Quit XXX" + str(self)

    def kill_world(self) -> None:
        self.world.killed()

    def manual_stop(self, kill_world: bool = True) -> None:
        if kill_world:
            self.kill_world()
        super().manual_stop()

    def manual_joint(self, address: Address) -> None:
        if not openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading and self.world.king:
            raise RuntimeError("Dynamic loading is not allowed!")

        if openfed.VERBOSE.is_verbose:
            logger.info(f"Manually add a new address: {repr(address)}")
        if self.world.king:
            self.pending_queue[str(address)] = [time.time(), 0, address]
        else:
            Joint(address, self.world)

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Maintainer",
            description=tablist(
                head=["Pending", "Finished", "Discard"],
                data=[len(self.pending_queue),
                      len(self.finished_queue),
                      len(self.discard_queue)]
            )
        )

    def __del__(self):
        del_maintainer_lock(self)
        super().__del__()

    def clear_finished_queue(self) -> None:
        self.finished_queue.clear()

    def clear_discard_queue(self) -> None:
        self.discard_queue.clear()
