from openfed.common import logger


def _auto_offline(func):
    """If any exception raised, we will offline the device and return False instead of original output.
    """

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.exception(e)
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
            logger.exception(e)
            return False
    return wrapper
