from loguru import \
    logger as \
    _logger  # do not import _logger in other modules directly in any time
from openfed.common.vars import DEBUG, VERBOSE


def log_to_file(filename: str):
    """Output log information to a file also.
    """
    _logger.add(filename)


def info(msg: str):
    if DEBUG.is_debug or VERBOSE.is_verbose:
        _logger.opt(depth=1).info(msg)


def success(msg: str):
    if DEBUG.is_debug or VERBOSE.is_verbose:
        _logger.opt(depth=1).success(msg)


def debug(msg: str):
    if DEBUG.is_debug:
        _logger.opt(depth=1).debug(msg)


def error(msg: str):
    _logger.opt(depth=1).error(msg)


def exception(msg: str):
    _logger.opt(depth=1).exception(msg)


def warning(msg: str):
    if DEBUG.is_debug:
        _logger.opt(depth=1).warning(msg)
