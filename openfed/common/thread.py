from abc import abstractmethod
from threading import Thread
from typing import Any, Dict

import openfed
import openfed.utils as utils
from openfed.common.logging import logger
from typing_extensions import final


class SafeTread(Thread):
    stopped: bool

    def __init__(self, daemon: bool = True):
        super().__init__(name="OpenFed SafeTread")
        self.stopped = False
        self.setDaemon(daemon)

        # register to global pool
        _thread_pool[self] = utils.time_string()

        if openfed.DEBUG.is_debug:
            logger.info(f"Create Thread: {self}")

    @final
    def run(self):
        """
            Implement safe_run() instead.
        """
        self.safe_exit(self.safe_run())

        self.stopped = True

    def safe_exit(self, msg: str):
        if openfed.DEBUG.is_debug:
            time_string = _thread_pool[self]
            logger.info((
                f"Exited Thread\n"
                f"{self}"
                f"{msg if msg else ''}\n"
                f"Created Time: {time_string}\n"
                f"Exited Time: {utils.time_string()}")
            )
        del _thread_pool[self]

    def __repr__(self) -> str:
        return "SafeTread"

    @abstractmethod
    def safe_run(self) -> Any:
        """Implement your method here.
        """

    def manual_stop(self):
        """Set stopped to True.
        """
        self.stopped = True


# Record global thread
_thread_pool:  Dict[SafeTread, str] = {}
