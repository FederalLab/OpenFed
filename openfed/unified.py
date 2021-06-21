# Provide a unified api for users on backend and frontend.
from openfed.backend import Backend
from openfed.frontend import Frontend


class API(Frontend, Backend):
    # A flag indicates whether this api is frontend or backend
    frontend: bool

    def __init__(self, frontend: bool = True):
        """
        Args:
            frontend: if ture, api act as a frontend. Otherwise backend.
        """
        self.frontend = frontend

    def _frontend_access(func):
        def wrapper(self, *args, **kwargs):
            if not self.frontend:
                raise RuntimeError("This function only used for frontend.")
            else:
                return func(self, *args, **kwargs)
        return wrapper

    def _backend_access(func):
        def wrapper(self, *args, **kwargs):
            if self.frontend:
                raise RuntimeError("This function only used for frontend.")
            else:
                return func(self, *args, **kwargs)
        return wrapper
