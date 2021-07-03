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

    def __init__(self):
        self.flag = False

    def set(self, flag: bool):
        self.flag = flag

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Vars",
            description="Base class for global variables."
        )


class _DAL(Vars):
    def __init__(self):
        self.flag = True

    def set_dal(self):
        self.flag = True

    def unset_dal(self):
        self.flag = False

    @property
    def is_dal(self) -> bool:
        return self.flag

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="DAL",
            description="If dal is enabled, a thread will be created to maintain new connection."
                        "Otherwise, OpenFed will get stuck until all address are correctly jointed."
                        "Call set_dal() to set dal, unset_dal to disable it."
        )


DAL = _DAL()


class _ASYNC_OP(Vars):
    def __init__(self):
        self.flag = True

    def set_async_op(self):
        self.flag = True

    def unset_async_op(self):
        self.flag = False

    @property
    def is_async_op(self) -> bool:
        return self.flag

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="ASYNC_OP",
            description="If True, the download and upload operation will return an handler."
                        "Otherwise, it will be blocked until finish."
        )


ASYNC_OP = _ASYNC_OP()
