from loguru import logger


def log_to_file(filename: str):
    logger.add(filename)