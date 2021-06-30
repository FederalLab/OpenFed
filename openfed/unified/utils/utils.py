from openfed.common.exception import AccessError
from openfed.common.logging import logger
from openfed.federated.utils import auto_filterout


def frontend_access(func):
    @auto_filterout
    def wrapper(self, *args, **kwargs):
        if not self.frontend:
            logger.debug(AccessError(func))
        else:
            return func(self, *args, **kwargs)
    return wrapper


def backend_access(func):
    @auto_filterout
    def wrapper(self, *args, **kwargs):
        if self.frontend:
            logger.debug(AccessError(func))
        else:
            return func(self, *args, **kwargs)
    return wrapper


def convert_to_list(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    else:
        return x


def after_connection(func):
    def wrapper(self, *args, **kwargs):
        if self.maintainer is None:
            raise AccessError(
                f"{func} only available after connection build.")
        else:
            return func(self, *args, **kwargs)
    return wrapper


def before_connection(func):
    def wrapper(self, *args, **kwargs):
        if not (self.maintainer is None):
            raise AccessError(
                f"{func} only available before connection build.")
        else:
            return func(self, *args, **kwargs)
    return wrapper
