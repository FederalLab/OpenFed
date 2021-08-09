import copy

from openfed.container import aggregators, reducers
from mmcv.utils import Registry, build_from_cfg

AGGREGATORS         = Registry('aggregator')
REDUCERS            = Registry('reducer')
AGGREGATOR_BUILDERS = Registry('aggregator builder')
REDUCER_BUILDERS    = Registry('reducer builder')


def register_aggregators():
    aggregator = []
    for _aggregator in aggregators:
        AGGREGATORS.register_module()(_aggregator)
        aggregator.append(_aggregator.__class__.__name__)
    return aggregator


def register_reducers():
    reducer = []
    for _reducer in reducers:
        REDUCERS.register_module()(_reducer)
        reducer.append(_reducer.__class__.__name__)
    return reducer


OPENFED_AGGREGATOR = register_aggregators()
OPENFED_REDUCER = register_reducers()


def build_aggregator_constructor(cfg):
    return build_from_cfg(cfg, AGGREGATOR_BUILDERS)


def build_reducer_constructor(cfg):
    return build_from_cfg(cfg, REDUCER_BUILDERS)


def build_aggregator(model, cfg):
    aggregator_cfg = copy.deepcopy(cfg)
    constructor_type = aggregator_cfg.pop('constructor',
                                          'DefaultAggregatorConstructor')
    aggregator_constructor = build_aggregator_constructor(
        dict(
            type=constructor_type,
            aggregator_cfg=aggregator_cfg,))
    aggregator = aggregator_constructor(model)
    return aggregator


def build_reducer(cfg):
    reducer_cfg = copy.deepcopy(cfg)
    constructor_type = reducer_cfg.pop('constructor',
                                       'DefaultReducerConstructor')
    reducer_constructor = build_reducer_constructor(
        dict(
            type=constructor_type,
            reducer_cfg=reducer_cfg,))
    reducer = reducer_constructor()
    return reducer
