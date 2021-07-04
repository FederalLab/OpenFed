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
from copy import deepcopy
from typing import Any, Dict, TypeVar

from openfed.utils import openfed_class_fmt, tablist

_T = TypeVar("_T", bound='TaskInfo')


class TaskInfo(object):
    r"""Provide same methods for better management of task info.
    """

    def __init__(self, ):
        super().__init__()

    @property
    def info_dict(self):
        info_dict = deepcopy(self.__dict__)
        json.dumps(info_dict)
        return info_dict

    def load_dict(self, o_dict: Dict[str, Any]) -> _T:
        self.__dict__.update(o_dict)
        return self

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="TaskInfo",
            description=tablist(
                head=list(self.info_dict.keys()),
                data=list(self.info_dict.values())
            )
        )
