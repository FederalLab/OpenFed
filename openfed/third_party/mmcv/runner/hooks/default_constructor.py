from mmcv.utils import build_from_cfg

from .builder import HOOK_BUILDERS, HOOKS


@HOOK_BUILDERS.register_module()
class DefaultHookConstructor:
    def __init__(self, hook_cfg):
        if not isinstance(hook_cfg, dict):
            raise TypeError('hook_cfg should be a dict',
                            f'but got {type(hook_cfg)}')
        self.hook_cfg = hook_cfg

    def __call__(self):
        hook_cfg = self.hook_cfg.copy()
        return build_from_cfg(hook_cfg, HOOKS)
