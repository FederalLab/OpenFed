from openfed.utils import openfed_class_fmt


class ConnectTimeout(Exception):
    """Timeout while building a new connection.
    """

    def __init__(self, msg=""):
        self.msg = msg

    def __str__(self):
        return openfed_class_fmt.format(
            class_name="ConnectTimeout",
            description=self.msg
        )
