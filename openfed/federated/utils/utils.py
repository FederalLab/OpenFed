import openfed
from openfed.common.logging import logger
from typing_extensions import final


def _auto_offline(func):
    """If any exception raised, we will offline the device and return False instead of original output.
    """

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if openfed.DEBUG.is_debug:
                raise e
            if openfed.VERBOSE.is_verbose:
                logger.error(e)
            # Set offline
            self.offline()
            return False
    return wrapper


def _auto_filterout(func):
    """If any exception raised, we will filter out those exception, and return False instead of original output.
    """

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if openfed.DEBUG.is_debug:
                raise e
            if openfed.VERBOSE.is_verbose:
                logger.warning(e)
            return False
    return wrapper
