from mmcv.utils import build_from_cfg

from .builder import (AGGREGATOR_BUILDERS, AGGREGATORS, REDUCER_BUILDERS,
                      REDUCERS)


@AGGREGATOR_BUILDERS.register_module()
class DefaultAggregatorConstructor:
    def __init__(self, aggregator_cfg):
        if not isinstance(aggregator_cfg, dict):
            raise TypeError('aggregator_cfg should be a dict',
                            f'but got {type(aggregator_cfg)}')
        self.aggregator_cfg = aggregator_cfg

    def __call__(self, model):
        aggregator_cfg = self.aggregator_cfg.copy()
        aggregator_cfg['params'] = model.parameters()
        return build_from_cfg(aggregator_cfg, AGGREGATORS)


@REDUCER_BUILDERS.register_module()
class DefaultReducerConstructor:
    def __init__(self, reducer_cfg):
        if not isinstance(reducer_cfg, dict):
            raise TypeError('reducer_cfg should be a dict',
                            f'but got {type(reducer_cfg)}')
        self.reducer_cfg = reducer_cfg

    def __call__(self):
        reducer_cfg = self.reducer_cfg.copy()
        return build_from_cfg(reducer_cfg, REDUCERS)
