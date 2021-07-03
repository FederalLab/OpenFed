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


from typing import List


class Wrapper(object):
    """Provide some method to wrap a class with support of Package.
    """
    pack_key_list: List = None
    unpack_key_list: List = None

    def add_pack_key(self, key: str):
        assert isinstance(key, str), "Only string format keys are supported."
        if self.pack_key_list is None:
            self.pack_key_list = []
        if key in self.pack_key_list:
            raise KeyError(f"Duplicate key: {key}.")
        self.pack_key_list.append(key)

    def add_unpack_key(self, key: str):
        assert isinstance(key, str), "Only string format keys are supported."
        if self.unpack_key_list is None:
            self.unpack_key_list = []
        if key in self.unpack_key_list:
            raise KeyError(f"Duplicate key: {key}.")
        self.unpack_key_list.append(key)

    def add_pack_key_list(self, keys: List[str]):
        [self.add_pack_key(key) for key in keys]

    def add_unpack_key_list(self, keys: List[str]):
        [self.add_unpack_key(key) for key in keys]
