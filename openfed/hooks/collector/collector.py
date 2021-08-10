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
 

import json
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Type

from openfed.common import logger
from openfed.utils import openfed_class_fmt

from ..hooks import Hooks


class Recorder(object):
    """Recorder makes sure that some collectors can only be created once. 
    """
    provided_collector_dict: Dict[str, Any] = dict()
    collector_pool: Dict[Any, Dict[Type, Any]] = defaultdict(dict)

    def __init__(self, obj: str, informer: Any):
        assert isinstance(obj, str)
        self.obj = obj
        self.informer = informer

    def __call__(self, *args, **kwargs):
        collectors = self.collector_pool[self.informer]

        if self.obj in self.provided_collector_dict:
            # get class by name.
            obj = self.provided_collector_dict[self.obj]
        else:
            logger.debug("Invalid collector.")
            return None
        if obj.bounding_name in collectors:
            return collectors[obj.bounding_name]
        else:
            logger.debug("Build a new collector.")
            ins = obj(*args, **kwargs)
            # put it into pool
            collectors[self.informer][obj.bounding_name] = ins
            return ins

    @classmethod
    def register(cls, obj: Any):
        if obj.bounding_name not in cls.provided_collector_dict:
            assert obj.bounding_name.startswith("Collector")
            logger.debug("Recorder collector %s" % obj.bounding_name)
            cls.provided_collector_dict[obj.bounding_name] = obj
        return obj

    @classmethod
    def add_to_pool(cls, func: Callable):
        def _add_to_pool(self, collector):
            # Recorder collector to collector pool
            cls.collector_pool[self][collector.bounding_name] = collector
            return func(self, collector)
        return _add_to_pool


@Recorder.register
class Collector(Hooks):
    """Some useful utilities to collect message.
    What's more, Collector also provide necessary function to better 
    visualizing the received messages.
    """
    # used to index or retrieve the message
    bounding_name: str = "Collector.base"

    # If True, run collect in leader
    leader_collector: bool = True
    # If True, run collect in follower
    follower_collector: bool = True

    # If True, scatter self message in leader
    leader_scatter: bool = True
    # If True, scatter self message in follower
    follower_scatter: bool = True

    def __init__(self, once_only: bool = False):
        super().__init__()
        # If True, this collector will only be called once.
        # Otherwise, it will be called everytime when the ends
        # want to upload/download data.
        self.once_only: bool = once_only
        # Only be used while once_only is True
        self.collected: bool = False
        self.scattered: bool = False

    @abstractmethod
    def collect(self) -> Any:
        """Implement related functions to collect messages.
        """

    def load_message(self, message: Any):
        self.message = message
        self.collected = True

    @abstractmethod
    def better_read(self) -> str:
        """Print a better string to visualize the received message.
        """

    def __call__(self) -> Any:
        output = self.collect()
        # check output automatically
        json.dumps(output)

        return output

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name=self.bounding_name,
            description=self.better_read()
        )


# trigger the auto register.
# NOTE: If the collector must be initialized with specified operation,
# do not register them here.
# Such as the lr_scheduler, which must be initialized at both ends.
Collector()
