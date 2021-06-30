from abc import abstractmethod
from threading import Lock
from typing import Union, List

from openfed.common import Clone


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


class Step(Clone):
    step_name: str

    def __init__(self):
        # automatically register step hooks to backend
        if len(_backend) > 0:
            _backend[0].register_step(self)

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> Union[None, bool]:
        ...


class MultiStep(Step):
    step_name: List[str] = []

    def _after_destroy(self):
        self.step_name.append('after_destroy')

    def _after_download(self):
        self.step_name.append('after_download')

    def _after_download(self):
        self.step_name.append('after_download')

    def _at_failed(self):
        self.step_name.append('at_failed')

    def _at_invalid_state(self):
        self.step_name.append('at_invalid_state')

    def _at_last(self):
        self.step_name.append('at_last')

    def _at_new_episode(self):
        self.step_name.append('at_new_episode')

    def _at_zombie(self):
        self.step_name.append('at_zombie')

    def _before_destory(self):
        self.step_name.append('before_destory')

    def _before_download(self):
        self.step_name.append('before_download')

    def _before_upload(self):
        self.step_name.append('before_upload')
