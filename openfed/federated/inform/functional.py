import json
import platform
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Type

import torch
from openfed.common import logger
from openfed.utils import openfed_class_fmt, tablist
from torch.optim.lr_scheduler import _LRScheduler


class Informer():
    ...


class Collector():
    ...


class Register(object):
    provided_collector_dict: Dict[str, Collector] = dict()
    collector_pool: Dict[Informer, Dict[Type, Collector]] = defaultdict(dict)

    def __init__(self, obj: str, informer: Informer):
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
            logger.debug("Load already exists collector.")
            return collectors[obj.bounding_name]
        else:
            logger.debug("Build a new collector.")
            ins = obj(*args, **kwargs)
            # put it into pool
            collectors[self.informer][obj.bounding_name] = ins
            return ins

    @classmethod
    def register(cls, obj: Collector):
        if obj.bounding_name not in cls.provided_collector_dict:
            assert obj.bounding_name.startswith("Collector")
            logger.info("Register collector %s" % obj.bounding_name)
            cls.provided_collector_dict[obj.bounding_name] = obj
        return obj

    @classmethod
    def add_to_pool(cls, func: Callable):
        def _add_to_pool(self, collector):
            # Register collector to collector pool
            cls.collector_pool[self][collector.bounding_name] = collector
            return func(self, collector)
        return _add_to_pool


@Register.register
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


# trigger the auto register.
# NOTE: If the collector must be initialized with specified operation,
# do not register them here.
# Such as the lr_scheduler, which must be initialized at both ends.
Collector()


@Register.register
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
                force_in_one_row=True,
            )


SystemInfo()


@Register.register
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
                device_properties=torch.cuda.get_device_properties(
                    torch.cuda.current_device()),
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


GPUInfo()


@Register.register
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
