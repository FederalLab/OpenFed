from abc import abstractmethod
from threading import Lock
from typing import List, Union

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
    # set when call hooks
    current_step: str

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
        """Call this if necessary at subclass init process."""
        self.step_name.append('after_destroy')

    def after_destroy(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _after_download(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('after_download')

    def after_download(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _after_upload(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('after_upload')

    def after_upload(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_failed(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('at_failed')

    def at_failed(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_invalid_state(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('at_invalid_state')

    def at_invalid_state(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_last(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('at_last')

    def at_last(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_new_episode(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('at_new_episode')

    def at_new_episode(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_zombie(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('at_zombie')

    def at_zombie(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _before_destory(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('before_destory')

    def before_destory(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _before_download(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('before_download')

    def before_download(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _before_upload(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('before_upload')

    def before_upload(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, backend: Backend, *args, **kwargs):
        if self.current_step in self.step_name:
            func = getattr(self, self.current_step)
            return func(self, backend, *args, **kwargs)
        else:
            return None
