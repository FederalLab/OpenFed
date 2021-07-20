# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys


class Logger(object):
    """A logger class that wrapper the loguru.
    """

    def __init__(self):
        from loguru import logger
        self.logger = logger
        # Remove the default logger.
        # The default logger will print log information
        # to screen directly.
        self.logger.remove()

    def log_level(self, level: str = "INFO"):
        # Add a new logger.
        self.logger.add(sys.stderr, level=level)
        self.level = level

    def log_to_file(self, filename: str, *args, **kwargs):
        self.logger.add(
            filename,
            *args,
            level=kwargs.get('level', self.level),
            **kwargs)

    def error(self, msg, depth=1):
        """
        .. Example::
            >>> logger.error("Error")
            2021-07-20 22:36:30.521 | ERROR    | __main__:<module>:1 - Error
        """
        self.logger.opt(depth=depth).error(msg)

    def warning(self, msg, depth=1):
        """
        .. Example::
            >>> logger.warning("Warning")
            2021-07-20 22:37:06.037 | WARNING  | __main__:<module>:1 - Warning
        """
        self.logger.opt(depth=depth).warning(msg)

    def info(self, msg, depth=1):
        """
        .. Example::
            >>> logger.info("Info")
            2021-07-20 22:37:44.523 | INFO     | __main__:<module>:1 - Info
        """
        self.logger.opt(depth=depth).info(msg)

    def debug(self, msg, depth=1):
        """
        .. Example::
            >>> logger.debug("Debug")
            2021-07-20 22:38:13.817 | DEBUG    | __main__:<module>:1 - Debug
        """
        self.logger.opt(depth=depth).debug(msg)

    def success(self, msg, depth=1):
        """
        .. Example::
            >>> logger.success("Success")
            2021-07-20 22:38:40.479 | SUCCESS  | __main__:<module>:1 - Success
        """
        self.logger.opt(depth=depth).success(msg)


logger = Logger()
