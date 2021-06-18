from loguru import logger
from .vars import DEBUG, VERBOSE


def _prefix(func):
    def wraper(msg):
        return func(f"<OpenFed> {msg}")
    return wraper


@_prefix
def log_debug_info(msg):
    if DEBUG:
        logger.debug(msg)


@_prefix
def log_verbose_info(msg):
    if VERBOSE:
        logger.info(msg)


def log_info(msg):
    log_debug_info(msg)
    log_verbose_info(msg)


@_prefix
def log_error_info(msg):
    logger.error(msg)


def log_to_file(filename: str):
    logger.add(filename)
