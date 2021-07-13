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


from typing import Any, Union

from openfed.utils import openfed_class_fmt


class OpenFedException(Exception):
    """A base class for exceptions.
    """

    def __init__(self,
                 exception_name: str = 'OpenFedException',
                 msg: Union[str,  Any] = ''):
        super().__init__()
        self.msg = msg
        self.exception_name = exception_name

    def __str__(self):
        return openfed_class_fmt.format(
            class_name=self.exception_name,
            description=self.msg
        )


class AccessError(OpenFedException):
    """If backend/frontend cross refer to each other functions, raised.
    """

    def __init__(self, msg: Union[str,  Any] = ''):
        super().__init__("AccessError", msg=msg)


class ConnectionNotBuild(OpenFedException):
    """Some operation only be available when the connection correctly built.
    """

    def __init__(self, msg: Union[str,  Any] = ''):
        super().__init__('ConnectionNotBuild', msg=msg)


class InvalidAddress(OpenFedException):
    """If not a valid address is given, raise it.
    """

    def __init__(self, msg: Union[str,  Any] = ''):
        super().__init__("InvalidAddress", msg=msg)


class ConnectTimeout(OpenFedException):
    """Timeout while building a new connection.
    """

    def __init__(self, msg: Union[str, Any] = ""):
        super().__init__("ConnectTimeout", msg=msg)


class InvalidStoreReading(OpenFedException):
    """Read an invalid store value.
    It mainly because the too many request for the single
    key value store at the same time.
    By default, we will return the cached one instead to 
    avoid this Error.
    """

    def __init__(self, msg: Union[str, Any] = ""):
        super().__init__("InvalidStoreReading", msg=msg)


class InvalidStoreWriting(OpenFedException):
    """Write store failed.
    It mainly because the too many request for the single
    key value store at the same time.
    By default, we will return the cached one instead to 
    avoid this Error.
    """

    def __init__(self, msg: Union[str, Any] = ""):
        super().__init__("InvalidStoreWriting", msg=msg)


class BuilddeliveryFailed(OpenFedException):
    """This Error is raised when we failed to other ends data at
    the initialization of Delivery.

    You can try to rebuild this delivery or just to discard this delivery.
    """

    def __init__(self, msg: Union[str, Any] = ""):
        super().__init__("BuilddeliveryFailed", msg=msg)


class DeviceOffline(OpenFedException):
    """This error is raised when client/server is offline.
    """

    def __init__(self, msg: Union[str, Any] = ""):
        super().__init__("DeviceOffline", msg=msg)


class WrongState(OpenFedException):
    """This error is raised when client in an unexpected state.
    """

    def __init__(self, msg: Union[str, Any] = ""):
        super().__init__("WrongState", msg=msg)
