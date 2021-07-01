from typing import Any, Dict

from openfed.utils import openfed_class_fmt, tablist


class TaskInfo(object):
    _info_dict: Dict[str, Any]

    def __init__(self, ):
        super().__init__()

        self._info_dict = {}

    def add_info(self, key: str, value: Any):
        self._info_dict[key] = value

    def get_info(self, key: str):
        return self._info_dict[key]

    def as_dict(self):
        return self._info_dict

    def load_dict(self, o_dict: Dict[str, Any]):
        self._info_dict.update(o_dict)

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="TaskInfo",
            description=tablist(
                head=list(self._info_dict.keys()),
                data=list(self._info_dict.values())
            )
        )
