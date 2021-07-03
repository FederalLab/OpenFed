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


import json
from typing import Any, Dict, TypeVar

from openfed.utils import openfed_class_fmt, tablist

from .logging import logger

_T = TypeVar("_T", bound='TaskInfo')


class TaskInfo(object):
    r"""Provide same methods for better management of task info.
    """
    _info_dict: Dict[str, Any]

    def __init__(self, ):
        super().__init__()

        self._info_dict = {}

    def set(self, key: str, value: Any):
        if key in self._info_dict:
            logger.debug(f"Reset {key} {self._info_dict[key]} as {value}")
        self._info_dict[key] = value
        # check value is valid or not
        json.dumps(self._info_dict)

    def get(self, key: str):
        return self._info_dict[key]

    @property
    def as_dict(self):
        return self._info_dict

    def load_dict(self, o_dict: Dict[str, Any]) -> _T:
        self._info_dict.update(o_dict)
        return self

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="TaskInfo",
            description=tablist(
                head=list(self._info_dict.keys()),
                data=list(self._info_dict.values())
            )
        )
