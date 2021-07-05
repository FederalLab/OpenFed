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


from openfed.common import logger
from ..step import AtLast


class Terminate(AtLast):
    max_version: int
    max_loop_times: int

    def __init__(self, max_loop_times: int = -1, max_version: int = -1):
        """
        Args:
            max_loop_times: if loop times exceed this number, we will stop the server.
            max_version: when inner version number achieves this number, we will stop server.
        """
        super().__init__()
        self.max_loop_times = max_loop_times
        self.max_version = max_version

    def step(self, backend, *args, **kwargs) -> None:
        if self.max_version != -1 and backend.version >= self.max_version:
            logger.info("Terminate! Max version achieves.")
            backend.manual_stop()
        if self.max_loop_times != -1 and backend.loop_times >= self.max_loop_times:
            logger.info("Terminate! Max loop times achieves.")
            backend.manual_stop()
