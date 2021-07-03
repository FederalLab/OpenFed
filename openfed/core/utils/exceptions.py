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


from openfed.common import OpenFedException
from openfed.utils import openfed_class_fmt


class ConnectTimeout(OpenFedException):
    """Timeout while building a new connection.
    """

    def __init__(self, msg: str = ""):
        super().__init__("ConnectTimeout", msg=msg)


class InvalidStoreReading(OpenFedException):
    """Read an invalid store value.
    It mainly because the too many request for the single
    key value store at the same time.
    By default, we will return the cached one instead to 
    avoid this Error.
    """

    def __init__(self, msg: str = ""):
        super().__init__("InvalidStoreReading", msg=msg)


class InvalidStoreWriting(OpenFedException):
    """Write store failed.
    It mainly because the too many request for the single
    key value store at the same time.
    By default, we will return the cached one instead to 
    avoid this Error.
    """

    def __init__(self, msg: str = ""):
        super().__init__("InvalidStoreWriting", msg=msg)


class BuildReignFailed(OpenFedException):
    """This Error is raised when we failed to other ends data at
    the initialization of Reign.

    You can try to rebuild this reign or just to discard this reign.
    """

    def __init__(self, msg: str = ""):
        super().__init__("BuildReignFailed", msg=msg)


class DeviceOffline(OpenFedException):
    """This error is raised when client/server is offline.
    """

    def __init__(self, msg: str = ""):
        super().__init__("DeviceOffline", msg=msg)


class WrongState(OpenFedException):
    """This error is raised when client in an unexpected state.
    """

    def __init__(self, msg: str = ""):
        super().__init__("WrongState", msg=msg)
