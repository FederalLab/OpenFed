import json
import platform
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Type, Union

import torch
from openfed.common import Clone, logger, peeper
from openfed.utils import openfed_class_fmt, tablist
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler


class Cypher(Clone):
    r"""Cypher: encrypt/decrypt data in pairs.
    The encrypt and decrypt functions will be called in two ends respectively.
    You can store the inner operation in the returned dictionary directly, but not 
    specify then as self.xxx=yyy.
    """

    def __init__(self):
        super().__init__()
        if peeper.api is not None:
            peeper.api.register_everything(self)

    def encrypt(self, key: Union[str, Tensor], value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package before transfer to the other end.
        """
        raise NotImplementedError(
            "You must implement the encrypt function for Cypher.")

    def decrypt(self, key: Union[str, Tensor], value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package received from the other end.
        """
        raise NotImplementedError(
            "You must implement the decrypt function for Cypher.")


class FormatCheck(Cypher):
    """Format Check.
    1. Convert `value` to {'param': value} if value is a Tensor.
    2. Align all other tensor in value to `param`'s device. 
    """

    def encrypt(self, key: Union[str, Tensor], value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert isinstance(key, Tensor)
        # Convert to dict
        if isinstance(value, Tensor):
            value = dict(param=value)
        assert isinstance(value, dict)

        # Align device
        for k, v in value.items():
            value[k] = v.cpu()
        return value

    def decrypt(self, key: Union[str, Tensor], value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert isinstance(key, Tensor)
        # Convert to dict
        if isinstance(value, Tensor):
            value = dict(param=value)
        assert isinstance(value, dict)

        # Align device
        for k, v in value.items():
            value[k] = v.to(key.device)
        return value


class Recoder(object):
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
            logger.debug("Recoder collector %s" % obj.bounding_name)
            cls.provided_collector_dict[obj.bounding_name] = obj
        return obj

    @classmethod
    def add_to_pool(cls, func: Callable):
        def _add_to_pool(self, collector):
            # Recoder collector to collector pool
            cls.collector_pool[self][collector.bounding_name] = collector
            return func(self, collector)
        return _add_to_pool


@Recoder.register
class Collector(Clone):
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
        # If True, this collector will only be called once.
        # Otherwise, it will be called everytime when the ends
        # want to upload/download data.
        self.once_only: bool = once_only
        # Only be used while once_only is True
        self.collected: bool = False
        self.scattered: bool = False

        if peeper.api is not None:
            peeper.api.register_everything(self)

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


@Recoder.register
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


@Recoder.register
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


@Recoder.register
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