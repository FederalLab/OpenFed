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

import platform
from typing import Any

from openfed.common import logger
from openfed.utils import tablist

from .collector import Collector, Recorder


@Recorder.register
class SystemInfo(Collector):
    """Collect some basic system info.
    """
    bounding_name: str = "Collector.SystemInfo"

    message: Any = None

    leader_collector: bool = True
    follower_collector: bool = False

    leader_scatter: bool = False
    follower_scatter: bool = True

    def __init__(self) -> None:
        super().__init__(True)

    def collect(self) -> Any:
        if self.scattered is False:
            self.scattered = True
            self.my_message = dict(
                system=platform.system(),
                platform=platform.system(),
                version=platform.version(),
                architecture=platform.architecture(),
                machine=platform.machine(),
                node=platform.node(),
                processor=platform.processor(),
            )
        return self.my_message

    def better_read(self):
        if self.message is None:
            logger.debug("Empty message received.")
            return ""
        else:
            return tablist(
                head=["System", "Platform", "Version",
                      "Architecture", 'Machine', "Node",
                      "Processor"],
                data=[self.message["system"],
                      self.message["platform"],
                      self.message["version"],
                      self.message["architecture"],
                      self.message["machine"],
                      self.message["node"],
                      self.message["processor"]],
                force_in_one_row=True,
            )


SystemInfo()
