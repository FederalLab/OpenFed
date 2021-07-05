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

from typing import Any, Dict

from openfed.utils import openfed_class_fmt


class Peeper(object):
    """
    Collect the state of whole openfed.
    """
    obj_item_mapping: Dict[str, Any]

    def __init__(self):
        self.obj_item_mapping = dict()

    def add_to_peeper(self, obj: Any, item: Any) -> None:
        if obj not in self.obj_item_mapping:
            self.obj_item_mapping[obj] = item

    def get_from_peeper(self, obj: Any) -> Any:
        return self.obj_item_mapping[obj]

    def remove_from_peeper(self, obj: Any) -> None:
        if obj in self.obj_item_mapping:
            del self.obj_item_mapping[obj]

    def __setattr__(self, name: str, value: Any):
        if name == "obj_item_mapping":
            super().__setattr__(name, value)
        self.add_to_peeper(name, value)

    def __getattribute__(self, name: str) -> Any:
        if name == "obj_item_mapping":
            return super().__getattribute__(name)
        if name in self.obj_item_mapping:
            return self.obj_item_mapping[name]
        return super().__getattribute__(name)

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name  = "Peeper",
            description = self.obj_item_mapping
        )


peeper = Peeper()
