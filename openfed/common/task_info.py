from typing import Any, Dict, TypeVar

from openfed.utils import openfed_class_fmt, tablist
import json

from .logging import logger

_T = TypeVar("_T", bound='TaskInfo')


class TaskInfo(object):
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
