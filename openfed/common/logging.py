import sys

from loguru import logger


def log_level(level: str = "INFO"):
    # remove default logger
    logger.remove(0)
    logger.add(sys.stderr, level=level)


def log_to_file(filename: str, *args, **kwargs):
    logger.add(filename, *args, **kwargs)
