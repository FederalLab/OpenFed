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


def _pop_items(kwargs, key, default_value):
    if key in kwargs:
        return kwargs.pop(key)
    else:
        return default_value


class Logger(object):
    def __init__(self, level: str = "INFO"):
        from loguru import logger
        self.logger = logger
        self.logger.remove()
        self.level = level

    def log_level(self, level: str = "INFO"):
        # remove default logger
        self.logger.add(sys.stderr, level=level)
        self.level = level

    def log_to_file(self, filename: str, *args, **kwargs):
        self.logger.add(
            filename, *args,
            level=_pop_items(kwargs,
                             'level', self.level),
            **kwargs)

    def error(self, *args, **kwargs):
        depth = _pop_items(kwargs, 'depth', 1)
        self.logger.opt(depth=depth).error(*args, **kwargs)

    def warning(self, *args, **kwargs):
        depth = _pop_items(kwargs, 'depth', 1)
        self.logger.opt(depth=depth).warning(*args, **kwargs)

    def info(self, *args, **kwargs):
        depth = _pop_items(kwargs, 'depth', 1)
        self.logger.opt(depth=depth).info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        depth = _pop_items(kwargs, 'depth', 1)
        self.logger.opt(depth=depth).debug(*args, **kwargs)


logger = Logger()
