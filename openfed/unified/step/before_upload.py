from openfed.common.logging import logger
from openfed.unified.step.base import Backend, Step


class BeforeUpload(Step):
    step_name = 'before_upload'

    def __call__(self, backend: Backend, *args, **kwargs) -> bool:

        # Check version requirements
        # upload is to check other download version.
        if backend.reign.download_version > backend.version:
            logger.warning(
                f"Version not aligned. (request @{backend.reign.download_version}, but @{backend.version}).")
            # Version is not satisfied.
            return False

        assert backend.optimizer
        assert backend.aggregator
        assert backend.state_dict

        # reset old state
        backend.reign.reset()

        # pack new data
        backend.reign.set_state_dict(backend.state_dict)
        for aggregator, optimizer in zip(backend.aggregator, backend.optimizer):
            backend.reign.pack_state(aggregator)
            backend.reign.pack_state(optimizer)

        return True
