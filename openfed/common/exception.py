from openfed.utils import openfed_class_fmt


class OpenFedException(Exception):
    """A base class for exceptions.
    """

    def __init__(self, msg: str = ""):
        self.msg = f"OpenFed Exception\n{msg}"

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="OpenFedException",
            description=self.msg
        )

class AccessError(OpenFedException):
    """If backend/frontend cross refer to each other functions, raised.
    """
    def __init__(self, msg: str = ""):
        self.msg = f"AccessError\n{msg}"

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="AccessError",
            description=self.msg
        )

class ConnectionNotBuild(OpenFedException):
    """Some operation only be available when the connection correctly built.
    """
    def __init__(self, msg: str = ""):
        self.msg = f"ConnectionNotBuild\n{msg}"
    
    def __str__(self):
        return openfed_class_fmt.format(
            class_name="ConnectionNotBuild",
            description=self.msg
        )