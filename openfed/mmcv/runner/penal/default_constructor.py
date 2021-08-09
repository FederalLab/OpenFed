from mmcv.utils import build_from_cfg

from .builder import PENALIZER_BUILDERS, PENALIZERS


@PENALIZER_BUILDERS.register_module()
class DefaultPenalizerConstructor:
    def __init__(self, penalizer_cfg):
        if not isinstance(penalizer_cfg, dict):
            raise TypeError('penalizer_cfg should be a dict',
                            f'but got {type(penalizer_cfg)}')
        self.penalizer_cfg = penalizer_cfg

    def __call__(self):
        penalizer_cfg = self.penalizer_cfg.copy()
        return build_from_cfg(penalizer_cfg, PENALIZERS)
