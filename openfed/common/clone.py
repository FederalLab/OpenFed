from typing import Any


class Clone(object):
    """Provide the methods to clone a class itself and return a new one.

    This is useful for copying hook functions between different classes.
    """

    def clone(self) -> Any:
        return type('Clone', self.__bases__, dict(self.__dict__))()
