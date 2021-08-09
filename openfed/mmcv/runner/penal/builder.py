import copy
from openfed.optim.penal import penalizers
from mmcv.utils import Registry, build_from_cfg

PENALIZERS = Registry('penalizer')
PENALIZER_BUILDERS = Registry('penalizer builder')


def register_penalizers():
    penalizer = []
    for _penalizer in penalizers:
        PENALIZERS.register_module()(_penalizer)
        penalizer.append(_penalizer.__class__.__name__)
    return penalizer


OPENFED_PENALIZERS = register_penalizers()


def build_penalizer_constructor(cfg):
    return build_from_cfg(cfg, PENALIZER_BUILDERS)


def build_penalizer(cfg):
    penalizer_cfg = copy.deepcopy(cfg)
    constructor_type = penalizer_cfg.pop('constructor',
                                         'DefaultPenalizerConstructor')
    penalizer_constructor = build_penalizer_constructor(
        dict(
            type=constructor_type,
            penalizer_cfg=penalizer_cfg,))
    penalizer = penalizer_constructor()
    return penalizer