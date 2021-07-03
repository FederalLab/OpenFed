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


from abc import abstractmethod
from threading import Thread
from typing import Any, Dict

from openfed.common import logger
from openfed.utils import openfed_class_fmt, tablist, time_string
from typing_extensions import final


class SafeTread(Thread):
    stopped: bool

    def __init__(self, daemon: bool = True):
        super().__init__(name="OpenFed SafeTread")
        self.stopped = False
        self.setDaemon(daemon)

        # register to global pool
        _thread_pool[self] = time_string()

    @final
    def run(self):
        """
            Implement safe_run() instead.
        """
        self.safe_exit(self.safe_run())

        self.stopped = True

    def safe_exit(self, msg: str):
        logger.debug("\n" +
                     tablist(
                         head=["Exited thread", "MSG",
                               "Create time", "Exited time"],
                         data=[self, msg, _thread_pool[self], time_string()],
                         force_in_one_row=True,
                     )
                     )
        del _thread_pool[self]

    def __str__(self) -> str:
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
