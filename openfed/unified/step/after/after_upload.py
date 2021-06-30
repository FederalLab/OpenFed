from openfed.common.logging import logger

from ..base import Backend, Step


class AfterUpload(Step):
    step_name = 'after_upload'

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug(f'Upload @{backend.version} to {backend.reign}')
