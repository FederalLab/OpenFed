import copy

from openfed.hooks import collectors, steps, cyphers
from mmcv.utils import Registry, build_from_cfg

HOOKS = Registry('openfed hook')
HOOK_BUILDERS = Registry('openfed hook builder')


def register_hooks():
    hook = []
    for _hook in collectors + steps + cyphers:
        HOOKS.register_module()(_hook)
        hook.append(_hook.__class__.__name__)
    return hook


OPENFED_HOOKS = register_hooks()


def build_hook_constructor(cfg):
    return build_from_cfg(cfg, HOOK_BUILDERS)


def build_hook(cfg):
    hook_cfg = copy.deepcopy(cfg)
    constructor_type = hook_cfg.pop('constructor',
                                    'DefaultHookConstructor')
    step_constructor = build_hook_constructor(
        dict(
            type     = constructor_type,
            hook_cfg = hook_cfg,
        )
    )
    step = step_constructor()
    return step
