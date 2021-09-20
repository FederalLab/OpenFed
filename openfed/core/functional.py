# Copyright (c) FederalLab. All rights reserved.
import warnings

from openfed.federated.exceptions import DeviceOffline


def fed_context(func):
    r'''A decorator that can be used to provide federated communication context.

    .. warning::

        This decorator intends to be used only for class which
        contains :attr:``pipe``.
    '''
    def _fed_context(self, *args, **kwargs):
        def safe_call(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DeviceOffline:
                warnings.warn(f'Failed to call {func}')
                return False

        if self.pipe.dist_props.lock.locked():
            return safe_call(self, *args, **kwargs)
        else:
            with self.pipe.dist_props:
                return safe_call(self, *args, **kwargs)

    return _fed_context
