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
