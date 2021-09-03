# Copyright (c) FederalLab. All rights reserved.
import warnings

from openfed.federated.exceptions import DeviceOffline


def fed_context(func):
    def _fed_context(self, *args, **kwargs):
        def safe_call(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DeviceOffline as e:
                warnings.warn(f"Failed to call {func}")
                return False

        if self.pipe.dist_props.lock.locked():
            return safe_call(self, *args, **kwargs)
        else:
            with self.pipe.dist_props:
                return safe_call(self, *args, **kwargs)

    return _fed_context
