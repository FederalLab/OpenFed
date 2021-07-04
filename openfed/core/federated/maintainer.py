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
from threading import Lock
from typing import Dict, List, Tuple, Union

import openfed
from openfed.common import (Address_, Array, SafeTread, load_address_from_file,
                            logger, remove_address_from_pool)
from openfed.utils import convert_to_list, openfed_class_fmt, tablist

from ..space import World
from ..utils.lock import add_mt_lock, del_maintainer_lock
from .joint import Joint


class Maintainer(Array, SafeTread):
    """
    Dynamic build the connection.
    """
    # unfinished address
    # Address_ -> [create time, try times]
    pending_queue: Dict[Address_, Tuple[float, int]]

    # finished address
    # Address_ -> [connection time, try times]
    finished_queue: Dict[Address_, Tuple[float, int]]

    # discard address
    # Address_ -> [discarded time, try times]
    discard_queue: Dict[Address_, Tuple[float, int]]

    mt_lock: Lock
    # The shared information among all country in this maintainer.
    world: World

    abnormal_exited: bool

    def __init__(self,
                 world: World,
                 address         : Union[Address_, List[Address_]] = None,
                 address_file    : str                             = None,
                 max_try_times   : int                             = 5,
                 interval_seconds: float                           = 10) -> None:
        """
            Only a single valid address is allowed in client.
        """
        self.address_file = address_file

        address_list       = convert_to_list(address)
        self.pending_queue = {address: [time.time(), 0]
                              for address in address_list} if address_list is not None else {}
        self.finished_queue  = dict()
        self.discard_queue   = dict()
        self.abnormal_exited = False

        self.max_try_times    = max_try_times
        self.interval_seconds = interval_seconds

        Array.__init__(self, self.pending_queue)

        self.mt_lock = Lock()
        add_mt_lock(self, self.mt_lock)

        self.world = world

        self.read_address_from_file()
        # call here
        SafeTread.__init__(self)

        if self.world.leader:
            self.start()
            if not openfed.DAL.is_dal:
                self.join()
                if self.abnormal_exited:
                    # Raise RuntimeError in the main thread.
                    raise RuntimeError(
                        "Errors occurred while building connection to new address.")
        else:
            assert len(self) == 1, "Only single address is allowed."
            address, (create_time, try_times) = self[0]
            Joint(address, self.world)
            del self.pending_queue[address]
            self.finished_queue[address] = [time.time(), try_times+1]

    def read_address_from_file(self) -> None:
        address_list = load_address_from_file(self.address_file)

        for address in address_list:
            if address in self.pending_queue:
                # Already in pending queue.
                ...
            elif address in self.finished_queue:
                # Already in finished_queue.
                ...
            elif address in self.discard_queue:
                logger.info(f"Invalid address: {address}.\nDiscarded!")
                # Remove this invalid address from address_pool
                remove_address_from_pool(address)
            else:
                # add address to pending queue
                self.pending_queue[address] = [time.time(), 0]

    def safe_run(self) -> str:
        while not self.stopped and self.world.ALIVE:
            # update pending list
            self.read_address_from_file()

            def try_now(last_time, try_times) -> bool:
                return False if (time.time() - last_time < self.interval_seconds) or try_times >= self.max_try_times else True

            for address, (last_time, try_times) in self:
                if try_now(last_time, try_times):
                    joint = Joint(address, self.world)
                    joint.join()
                    if joint.build_success:
                        self.finished_queue[address] = [
                            time.time(), try_times + 1]
                        del self.pending_queue[address]
                    else:
                        try_times += 1
                        if try_times > self.max_try_times:
                            # Move to discard queue
                            self.discard_queue[address] = [
                                time.time(), try_times]
                            del self.pending_queue[address]
                            break
                        else:
                            self.pending_queue[address] = [
                                time.time(), try_times]

            if len(self) == 0:
                if openfed.DAL.is_dal:
                    time.sleep(self.interval_seconds)
                else:
                    break
            else:
                if not openfed.DAL.is_dal:
                    if len(self.discard_queue) != 0:
                        self.abnormal_exited = True
                        break
                else:
                    time.sleep(self.interval_seconds)

        return f"Build connection to {len(self.finished_queue)} addresses."

    def kill_world(self) -> None:
        self.world.killed()

    def manual_stop(self, kill_world: bool = True) -> None:
        if kill_world:
            self.kill_world()
        super().manual_stop()

    def manual_joint(self, address: Address_) -> None:
        if not openfed.DAL.is_dal and self.world.leader:
            raise RuntimeError("Dynamic Address Loading (ADL) is disabled.")

        if self.world.leader:
            self.pending_queue[address] = [time.time(), 0]
        else:
            Joint(address, self.world)

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name  = "Maintainer",
            description = tablist(
                head = ["Pending", "Finished", "Discard"],
                data = [len(self.pending_queue),
                      len(self.finished_queue),
                      len(self.discard_queue)],
                force_in_one_row = True,
            )
        )

    def __del__(self) -> None:
        del_maintainer_lock(self)
        super().__del__()

    def clear_queue(self) -> None:
        """Clear all address in queue.
        """
        self.finished_queue.clear()
        self.discard_queue.clear()
        self.pending_queue.clear()
