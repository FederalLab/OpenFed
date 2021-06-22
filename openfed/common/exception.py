from openfed.utils import openfed_class_fmt


class OpenFedException(Exception):
    """A base class for exceptions.
    """

    def __init__(self, msg: str = ""):
        self.msg = "OpenFed Eception" if not msg else msg

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="OpenFedException",
            description=self.msg
        )


class ConnectTimeout(OpenFedException):
    """Timeout while building a new connection.
    """

    def __init__(self, msg: str = ""):
        self.msg = "Waiting to beckend response, timeout." if not msg else msg

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
        self.msg = "Read an invalid store value." if not msg else msg

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="InvalidStoreReading",
            description=self.msg
        )


class BuildReignFailed(OpenFedException):
    """This Error is raised when we failed to other ends data at
    the initialization of Reign.

    You can try to rebuild this reign or just to discard this reign.
    """

    def __init__(self, msg: str = ""):
        self.msg = "Build Reign failed" if not msg else msg

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="BuildReignFailed",
            description=self.msg
        )
