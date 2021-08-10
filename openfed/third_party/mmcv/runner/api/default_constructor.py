from mmcv.utils import build_from_cfg

from .builder import API_BUILDERS, APIS


@API_BUILDERS.register_module()
class DefaultAPIConstructor:
    def __init__(self, api_cfg):
        if not isinstance(api_cfg, dict):
            raise TypeError('api_cfg should be a dict',
                            f'but got {type(api_cfg)}')
        self.api_cfg = api_cfg

    def __call__(self, world, state_dict, fed_optim, container):
        api_cfg = self.api_cfg.copy()
        api_cfg['world']      = world
        api_cfg['state_dict'] = state_dict
        api_cfg['fed_optim']  = fed_optim
        api_cfg['container']  = container

        return build_from_cfg(api_cfg, APIS)
