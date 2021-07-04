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


from openfed.utils import openfed_class_fmt


class Vars(object):
    flag: bool
    class_name: str

    def __init__(self, flag: bool, class_name: str):
        self.flag       = flag
        self.class_name = class_name

    def set(self, flag: bool):
        self.flag = flag

    def __str__(self):
        return openfed_class_fmt.format(
            class_name  = self.class_name,
            description = f"Flag: {self.flag}."
        )


class _DAL(Vars):
    """Dynamic Address Loading.
    """

    def __init__(self):
        super().__init__(True, 'DynamicAddressLoading')

    def set_dal(self) -> None:
        self.set(True)

    def unset_dal(self) -> None:
        self.set(False)

    @property
    def is_dal(self) -> bool:
        return self.flag


DAL = _DAL()


class _ASYNC_OP(Vars):
    """ASYNC OP.
    """

    def __init__(self):
        super().__init__(True, "ASYNC_OP")

    def set_async_op(self) -> None:
        self.set(True)

    def unset_async_op(self) -> None:
        self.set(False)

    @property
    def is_async_op(self) -> bool:
        return self.flag


ASYNC_OP = _ASYNC_OP()
