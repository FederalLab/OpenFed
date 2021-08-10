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


from typing import Dict

from torch.optim.lr_scheduler import _LRScheduler

from .collector import Collector, Recorder


@Recorder.register
class LRTracker(Collector):
    """Keep tack of learning rate during training.
    """
    bounding_name: str = "Collector.LRTracker"
    leader_collector: bool = False
    follower_collector: bool = True

    leader_scatter: bool = True
    follower_scatter: bool = False

    def __init__(self, lr_scheduler: _LRScheduler):
        super().__init__(False)
        self.lr_scheduler = lr_scheduler

    def collect(self) -> Dict:
        return self.lr_scheduler.state_dict()

    def load_message(self, message: Dict):
        self.lr_scheduler.load_state_dict(message)

    def better_read(self):
        return (
            "Lastest Learing Rate: "
            f"{self.lr_scheduler.get_last_lr()[0]}"
        )
