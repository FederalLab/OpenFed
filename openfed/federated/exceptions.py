# @Author            : FederalLab
# @Date              : 2021-09-25 16:52:22
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:52:22
# Copyright (c) FederalLab. All rights reserved.


class DeviceOffline(Exception):
    """Raises an exception if the device is offline.

    Example::

        >>> raise DeviceOffline()
        Traceback (most recent call last):
        File '<stdin>', line 1, in <module>
        openfed.common.exceptions.DeviceOffline
    """
