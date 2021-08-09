import copy
from typing import Dict, Union

import torch
from mmcv.utils import Registry, build_from_cfg
from openfed.api import API
from openfed.container import Container
from openfed.core import World
from openfed.optim import FedOptim

APIS = Registry('api')
API_BUILDERS = Registry('api builder')


def register_apis():
    """Register all default API from openfed to mmcv.
    """
    api = []

    APIS.register_module()(API)
    api.append("API")
    return api


OPENFED_APIS = register_apis()


def build_api_constructor(cfg):
    return build_from_cfg(cfg, API_BUILDERS)


def build_api(
        world: World,
        state_dict: Dict[str, torch.Tensor],
        fed_optim: FedOptim,
        container: Union[Container, None],
        cfg: Dict):

    api_cfg = copy.deepcopy(cfg)
    constructor_type = api_cfg.pop('constructor',
                                   'DefaultAPIConstructor')
    api_constructor = build_api_constructor(
        dict(
            type=constructor_type,
            api_cfg=api_cfg,
        )
    )
    api = api_constructor(
        world,
        state_dict,
        fed_optim,
        container)
    return api
