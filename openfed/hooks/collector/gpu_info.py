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


from typing import Any

import torch
from openfed.common import logger
from openfed.utils import tablist

from .collector import Collector, Recorder


@Recorder.register
class GPUInfo(Collector):
    """Collect some basic GPU information if GPU is available.
    """
    bounding_name: str = "Collector.GPUInfo"

    message: Any = None
    leader_collector: bool = True
    follower_collector: bool = False

    leader_scatter: bool = False
    follower_scatter: bool = True

    def __init__(self) -> None:
        super().__init__(True)

    def collect(self) -> Any:
        if self.scattered is False:
            if torch.cuda.is_available():
                self.my_message = dict(
                    device_count=torch.cuda.device_count(),
                    arch_list=torch.cuda.get_arch_list(),
                    device_capability=torch.cuda.get_device_capability(),
                    device_name=torch.cuda.get_device_name(),
                    current_device=torch.cuda.current_device(),
                )
            else:
                self.my_message = None
        return self.my_message

    def better_read(self):
        if self.message is None:
            logger.debug("Empty message received.")
            return ""
        else:
            return tablist(
                head=["Count", "Arch", "Capability",
                      "Name", "Current"],
                data=[self.message['device_count'],
                      self.message['arch_list'],
                      self.message['device_capability'],
                      self.message['device_name'],
                      self.message['current_device']],
                force_in_one_row=True
            )


GPUInfo()
