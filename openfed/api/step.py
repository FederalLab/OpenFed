from threading import Lock
from typing import List, Union

from openfed.common import Clone

from enum import Enum, unique


@unique
class StepName(Enum):
    # After
    AFTER_DESTROY = 'AFTER_DESTROY'
    AFTER_DOWNLOAD = 'AFTER_DOWNLOAD'
    AFTER_UPLOAD = 'AFTER_UPLOAD'

    # At
    AT_FIRST = "AT_FIRST"
    AT_FAILED = 'AT_FAILED'
    AT_INVALID_STATE = 'AT_INVALID_STATE'
    AT_LAST = 'AT_LAST'
    AT_NEW_EPISODE = 'AT_NEW_EPISODE'
    AT_ZOMBIE = 'AT_ZOMBIE'

    # Before
    BEFORE_DESTROY = 'BEFORE_DESTROY'
    BEFORE_DOWNLOAD = 'BEFORE_DOWNLOAD'
    BEFORE_UPLOAD = 'BEFORE_UPLOAD'


after_destroy = StepName.AFTER_DESTROY.value
after_download = StepName.AFTER_DOWNLOAD.value
after_upload = StepName.AFTER_UPLOAD.value

at_first = StepName.AT_FIRST.value
at_failed = StepName.AT_FAILED.value
at_invalid_state = StepName.AT_INVALID_STATE.value
at_last = StepName.AT_LAST.value
at_new_episode = StepName.AT_NEW_EPISODE.value
at_zombie = StepName.AT_ZOMBIE.value

before_destroy = StepName.BEFORE_DESTROY.value
before_download = StepName.BEFORE_DOWNLOAD.value
before_upload = StepName.BEFORE_UPLOAD.value


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
    step_name: StepName

    def __init__(self):
        # automatically register step hooks to backend
        if len(_backend) > 0:
            _backend[0].register_step(self)

    def __call__(self, backend: Backend, *args, **kwargs) -> Union[None, bool]:
        return self.step(backend, *args, **kwargs)

    def step(self, backend: Backend, *args, **kwargs) -> Union[None, bool]:
        raise NotImplementedError


class MultiStep(Step):
    step_name: List[StepName] = []

    def _after_destroy(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(after_destroy)

    def after_destroy(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _after_download(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(after_download)

    def after_download(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _after_upload(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append('after_upload')

    def after_upload(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_failed(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_failed)

    def at_failed(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_invalid_state(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_invalid_state)

    def at_invalid_state(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_last(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_last)

    def at_last(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_new_episode(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_new_episode)

    def at_new_episode(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _at_zombie(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(at_zombie)

    def at_zombie(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _before_destroy(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(before_destroy)

    def before_destroy(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _before_download(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(before_download)

    def before_download(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def _before_upload(self):
        """Call this if necessary at subclass init process."""
        self.step_name.append(before_upload)

    def before_upload(self, backend: Backend, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, backend: Backend, *args, **kwargs):
        if backend.current_step in self.step_name:
            func = getattr(self, backend.current_step.lower())
            return func(backend, *args, **kwargs)
        else:
            return None
