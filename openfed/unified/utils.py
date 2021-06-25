from loguru import logger
from openfed.common.exception import AccessError
from openfed.federated.utils.utils import _auto_filterout


def _frontend_access(func):
    @_auto_filterout
    def wrapper(self, *args, **kwargs):
        if not self.frontend:
            logger.debug(AccessError(func))
        else:
            return func(self, *args, **kwargs)
    return wrapper


def _backend_access(func):
    @_auto_filterout
    def wrapper(self, *args, **kwargs):
        if self.frontend:
            logger.debug(AccessError(func))
        else:
            return func(self, *args, **kwargs)
    return wrapper
