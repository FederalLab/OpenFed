import json
import platform
from abc import abstractmethod
from typing import Any, Dict

import openfed
from torch.optim.lr_scheduler import _LRScheduler

from ...common import logger


class Collector(object):
    """Some useful utilities to collect message.
    What's more, Collector also provide necessary funciton to better 
    visualizing the received messages.
    """
    # used to index or retrieve the message
    bounding_name: str = "Collector.base"

    @abstractmethod
    def collect(self) -> Any:
        """Implement related functions to collect messages.
        """

    @abstractmethod
    def load_message(self, message: Any):
        """Load the received message to inner state for better visualizing.
        """

    @abstractmethod
    def better_read(self) -> str:
        """Print a better string to visualize the received message.
        """

    def __call__(self) -> Any:
        output = self.collect()
        # check ouput automatically
        json.dumps(output)

        return output

    def __repr__(self) -> str:
        return self.better_read()


class SystemInfo(Collector):
    """Collect same basic system info.
    """
    bounding_name: str = "Collector.SystemInfo"

    message: Any = None

    def collect(self) -> Dict[str, str]:
        return dict(
            system=platform.system(),
            platform=platform.system(),
            version=platform.version(),
            architecture=platform.architecture(),
            machine=platform.machine(),
            node=platform.node(),
            processor=platform.processor(),
        )

    def load_message(self, message: Any):
        self.message = message

    def better_read(self):
        if self.message is None:
            if openfed.DEBUG.is_debug:
                logger.error("Empty message received.")
            return ""
        else:
            return (
                f"System Information List\n"
                f"System: {self.message['system']}\n"
                f"Platform: {self.message['platform']}\n"
                f"Version: {self.message['version']}\n"
                f"Architecture: {self.message['architecture']}\n"
                f"Machine: {self.message['machine']}\n"
                f"Node: {self.message['node']}\n"
                f"Processor: {self.message['processor']}\n"
            )


class LRTracker(Collector):
    """Keep tack of learning rate during training.
    """
    bounding_name: str = "Collector.LRTracker"

    def __init__(self, lr_scheduler: _LRScheduler):
        self.lr_scheduler = lr_scheduler

    def collect(self) -> Dict:
        return self.lr_scheduler.state_dict()

    def load_message(self, message: Dict):
        self.lr_scheduler.load_state_dict(message)

    def better_read(self):
        return (
            "Lastest Learing Rate\n"
            f"{self.lr_scheduler.get_last_lr()}"
        )


provided_collector_dict = {
    Collector.bounding_name: Collector,
    SystemInfo.bounding_name: SystemInfo,
    LRTracker.bounding_name: LRTracker,
}
