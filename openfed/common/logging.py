from loguru import logger


def log_to_file(filename: str):
    """Output log information to a file also.
    """
    logger.add(filename)
