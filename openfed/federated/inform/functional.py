import json
import platform
from abc import abstractmethod
from typing import Any, Dict

import openfed
import torch
import openfed.common.logging as logger
from openfed.utils import openfed_class_fmt
from openfed.utils.table import tablist
from torch.optim.lr_scheduler import _LRScheduler


class Collector(object):
    """Some useful utilities to collect message.
    What's more, Collector also provide necessary function to better 
    visualizing the received messages.
    """
    # used to index or retrieve the message
    bounding_name: str = "Collector.base"

    @abstractmethod
    def collect(self) -> Any:
        """Implement related functions to collect messages.
        """

    def load_message(self, message: Any):
        self.message = message

    @abstractmethod
    def better_read(self) -> str:
        """Print a better string to visualize the received message.
        """

    def __call__(self) -> Any:
        output = self.collect()
        # check output automatically
        json.dumps(output)

        return output

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name=self.bounding_name,
            description=self.better_read()
        )


class SystemInfo(Collector):
    """Collect some basic system info.
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

    def better_read(self):
        if self.message is None:
            logger.error("Empty message received.")
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
                items_per_row=4
            )


class GPUInfo(Collector):
    """Collect some basic GPU information if GPU is available.
    """
    bounding_name: str = "Collector.GPUInfo"

    message: Any = None

    def collect(self) -> Dict[str, str]:
        if torch.cuda.is_available():
            return dict(
                device_count=torch.cuda.device_count(),
                arch_list=torch.cuda.get_arch_list(),
                device_capability=torch.cuda.get_device_capability(),
                device_name=torch.cuda.get_device_name(),
                device_properties=torch.cuda.get_device_properties(),
                current_device=torch.cuda.current_device(),
            )
        else:
            return None

    def better_read(self):
        if self.message is None:
            logger.error("Empty message received.")
            return ""
        else:
            return tablist(
                head=["Count", "Arch", "Capability",
                      "Name", 'Properties', "Current"],
                data=[self.message['device_count'],
                      self.message['arch_list'],
                      self.message['device_capability'],
                      self.message['device_name'],
                      self.message['device_properties'],
                      self.message['current_device']],
                force_in_one_row=True
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
    GPUInfo.bounding_name: GPUInfo,
    LRTracker.bounding_name: LRTracker,
}
