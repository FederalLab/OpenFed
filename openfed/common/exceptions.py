# Copyright (c) FederalLab. All rights reserved.
from typing import Any, Union

from openfed.utils import openfed_class_fmt


class OpenFedException(Exception):
    """The base class for inner exceptions on OpenFed.
    """
    def __init__(self,
                 exception_name: str = 'OpenFedException',
                 msg: Union[str, Any] = ''):
        super().__init__()
        self.msg = msg
        self.exception_name = exception_name

    def __str__(self):
        return openfed_class_fmt.format(class_name=self.exception_name,
                                        description=self.msg)


class DeviceOffline(OpenFedException):
    """Raised when other end is offline (both for leader and follower).
    """
    def __init__(self, msg: Union[str, Any] = ""):
        super().__init__("DeviceOffline", msg=msg)
