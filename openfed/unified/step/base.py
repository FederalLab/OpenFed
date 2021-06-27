from abc import abstractmethod
from threading import Lock
from typing import Union


class Backend(object):
    ...


_step_lock = Lock()
_backend: Backend = []


class StepAt(object):
    """Get a lock, and register all step hooks into _backend automatically.
    """

    def __init__(self, backend: Backend):
        self.backend = backend

    def __enter__(self):
        _step_lock.acquire()
        assert len(_backend) == 0, "Forget to remove backend from last lock."
        _backend.append(self.backend)

    def __exit__(self, *unused):
        _backend.pop()
        assert len(_backend) == 0
        _step_lock.release()


class Step(object):
    step_name: str

    def __init__(self):
        # automatically register step hooks to backend
        if len(_backend) > 0:
            _backend[0].register_step(self)

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> Union[None, bool]:
        ...
