from typing import Any, Dict, List

import openfed
import torch.distributed as dist
from mmcv.runner import build_optimizer
from mmcv.utils import Registry, build_from_cfg
from openfed.container import build_container
from openfed.core import is_leader, leader
from openfed.optim import build_fed_optim

from .container import build_aggregator, build_reducer
from .hooks import build_hook
from .penal import build_penalizer
from .api import build_api

OPENFED = Registry('openfed')


@OPENFED.register_module()
class OpenFed(object):
    rank: int  # assign from runner
    model: Any  # assign from runner
    world_size: int  # assign from runner

    def __init__(
        self,
        model,
        optimizer,
        rank: int = -1,
        address_file: str = '',
        address_cfg: Dict[str, Any] = dict(
            backend='gloo',
            init_method='env:///',
            group_name='',
        ),
        leader_optimizer: Dict[str, Any] = dict(
            type='SGD',
            lr=1.0,
            momentum=0.9,
            weight_decay=0.0001,
        ),
        penalizer_cfg_list: List[Dict[str, Any]] = [],
        aggregator_cfg: Dict[str, Any] = dict(
            other_keys=[],
        ),
        reducer_cfg: Dict[str, Any] = dict(
        ),
        world_cfg: Dict[str, Any] = dict(
            role=leader,
            async_op='auto',
            dal=True,
            mtt=5,
        ),
        hook_cfg_list: List[Dict[str, Any]] = [],
        api_cfg: Dict[str, Any] = dict(),
    ):
        super().__init__()
        if rank != 0:
            # It is not necessary to build openfed core if rank > 0
            return

        role = world_cfg["role"]

        # Build fed optimizer
        if is_leader(role):
            optimizer = build_optimizer(model, leader_optimizer)

        for cfg in penalizer_cfg_list:
            cfg['role'] = role

        if aggregator_cfg['type'] == "ElasticAgg":
            contains_elastic_penalizer = False
            for cfg in penalizer_cfg_list:
                if cfg['type'] == 'ElasticPenalizer':
                    contains_elastic_penalizer = True
                    break
            else:
                assert contains_elastic_penalizer, "You must specify the `ElasticPenalizer` for `ElasticAgg`"

        if len(penalizer_cfg_list) == 0:
            penalizer = None
        else:
            penalizer = openfed.optim.PenalizerList(
                [build_penalizer(cfg) for cfg in penalizer_cfg_list]
            )

        fed_optim = build_fed_optim(optimizer, penalizer)

        # Build Container
        if is_leader(role):
            aggregator = build_aggregator(model, aggregator_cfg)
            reducer = build_reducer(reducer_cfg)
            container = build_container(aggregator, reducer)
        else:
            container = None

        # Build World
        world = openfed.core.World(**world_cfg)

        # Build OpenFed API
        openfed_api = build_api(world, model.state_dict(
            keep_vars=True), fed_optim, container, api_cfg)

        # Register step function
        if is_leader(role):
            with openfed_api:
                [build_hook(cfg) for cfg in hook_cfg_list]

        # Build Address
        address = openfed.build_address(
            rank=openfed.core.leader_rank if is_leader(role) else openfed.core.follower_rank,
            world_size=2,
            **address_cfg
        )

        # Connect to other end
        openfed_api.build_connection(
            address=address, address_file=address_file)

        self.openfed_api = openfed_api
        # Replace the original optimizer with pipe
        self.optimizer = fed_optim

    def train(self, to: bool, data_loader, **kwargs):
        if self.rank == 0:
            # Sync model with other end
            self.openfed_api.transfer(to=to)

        if not to:  # Download a model from the other end.
            # Broadcast new model to all nodes in distributed learning.
            if self.world_size > 1:
                for p in self.model.parameters():
                    dist.broadcast(p, src=0)

            # Accumulate gradient
            self.optimizer.acg(self.model, data_loader)


def build_openfed(cfg, default_args=None) -> OpenFed:
    return build_from_cfg(cfg, OPENFED, default_args=default_args)
