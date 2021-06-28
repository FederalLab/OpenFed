from openfed.common import OpenFedException
from openfed.utils import openfed_class_fmt


class ConnectTimeout(OpenFedException):
    """Timeout while building a new connection.
    """

    def __init__(self, msg: str = ""):
        self.msg = f"Connecting, timeout.\n{msg}"

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="ConnectTimeout",
            description=self.msg
        )


class InvalidStoreReading(OpenFedException):
    """Read an invalid store value.
    It mainly because the too many request for the single
    key value store at the same time.
    By default, we will return the cached one instead to 
    avoid this Error.
    """

    def __init__(self, msg: str = ""):
        self.msg = f"Read store failed.\n{msg}"

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="InvalidStoreReading",
            description=self.msg
        )


class InvalidStoreWriting(OpenFedException):
    """Write store failed.
    It mainly because the too many request for the single
    key value store at the same time.
    By default, we will return the cached one instead to 
    avoid this Error.
    """

    def __init__(self, msg: str = ""):
        self.msg = f"Write store failed.\n{msg}"

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="InvalidStoreWriting",
            description=self.msg
        )


class BuildReignFailed(OpenFedException):
    """This Error is raised when we failed to other ends data at
    the initialization of Reign.

    You can try to rebuild this reign or just to discard this reign.
    """

    def __init__(self, msg: str = ""):
        self.msg = f"Build Reign failed.\n{msg}"

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="BuildReignFailed",
            description=self.msg
        )


class DeviceOffline(OpenFedException):
    """This error is raised when client/server is offline.
    """

    def __init__(self, msg: str = ""):
        self.msg = f"Device offline.\n{msg}"

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="DeviceOffline",
            description=self.msg
        )


class WrongState(OpenFedException):
    """This error is raised when client in an unexpected state.
    """

    def __init__(self, msg: str = ""):
        self.msg = f"Wrong state.\n{msg}"

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="WrongState",
            description=self.msg
        )
