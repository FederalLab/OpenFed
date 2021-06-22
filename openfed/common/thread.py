from abc import abstractmethod
from threading import Thread
from typing import Any, Dict

import openfed
from openfed.common.logging import logger
from openfed.utils import openfed_class_fmt, time_string
from openfed.utils.table import tablist
from typing_extensions import final


class SafeTread(Thread):
    stopped: bool

    def __init__(self, daemon: bool = True):
        super().__init__(name="OpenFed SafeTread")
        self.stopped = False
        self.setDaemon(daemon)

        # register to global pool
        _thread_pool[self] = time_string()

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
            create_time_string = _thread_pool[self]
            logger.info(
                tablist(
                    head=["Exited Thread", "MSG",
                          "Created Time", "Exited Time"],
                    data=[self, msg, create_time_string, time_string()]
                )
            )
        del _thread_pool[self]

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="SafeThread",
        )

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
