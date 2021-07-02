import sys


class Logger(object):
    def __init__(self):
        from loguru import logger
        self.logger = logger
        self.logger.remove()

    def log_level(self, level: str = "INFO"):

        # remove default logger
        self.logger.add(sys.stderr, level=level)

    def log_to_file(self, filename: str, *args, **kwargs):
        self.logger.add(filename, *args, **kwargs)

    def error(self, *args, **kwargs):
        depth = kwargs.pop('depth') if 'depth' in kwargs else 1
        self.logger.opt(depth=depth).error(*args, **kwargs)

    def warning(self, *args, **kwargs):
        depth = kwargs.pop('depth') if 'depth' in kwargs else 1
        self.logger.opt(depth=depth).warning(*args, **kwargs)

    def info(self, *args, **kwargs):
        depth = kwargs.pop('depth') if 'depth' in kwargs else 1
        self.logger.opt(depth=depth).info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        depth = kwargs.pop('depth') if 'depth' in kwargs else 1
        self.logger.opt(depth=depth).debug(*args, **kwargs)


logger = Logger()
