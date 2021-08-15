from typing import Any, Dict, List
from warnings import warn

import openfed
import torch.distributed as dist
from mmcv.utils import Registry, build_from_cfg
from openfed.core import is_follower, is_leader, leader
from openfed.tools import build_optim

from .hooks import build_hook

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
        role: str = leader,
        address_file: str = '',
        fed_optim_cfg: Dict[str, Any] = dict(),
        hook_cfg_list: List[Dict[str, Any]] = []
    ):
        super().__init__()
        if rank != 0:
            # It is not necessary to build openfed core if rank > 0
            return

        if len(hook_cfg_list) == 0:
            hook_cfg_list = [
                dict(
                    type        = 'Aggregate',
                    count       = dict(train=-1),
                    checkpoint  = None,
                    max_version = -1,
                )
            ]
        
        if len(fed_optim_cfg) == 0:
            fed_optim_cfg = dict(
                type = 'fedavg',
                lr   = 1.0,
            )

        # Build fed optimizer
        if is_follower(role):
            # if this process is a client, we will use the passed in optimizer
            # else we will build a new optimizer for it.
            fed_optim_cfg['optimizer'] = optimizer
        fed_optim_cfg['role'] = role
        optimizer, aggregator = build_optim(
            fed_optim_cfg.pop('type'), model.parameters(), **fed_optim_cfg)

        # Build World
        world = openfed.core.World(role=role, mtt=15)

        # Build OpenFed API
        openfed_api = openfed.API(
            world,
            model.state_dict(keep_vars=True),
            optimizer,
            aggregator)

        # Register step function
        if is_leader(role):
            with openfed_api:
                aggregate_in = False
                for cfg in hook_cfg_list:
                    if cfg['type'] == 'Aggregate':
                        aggregate_in = True
                    build_hook(cfg)
                if not aggregate_in:
                    warn(
                        "Aggregate step function is not included in the hook config, which may cause a error.")

        # Connect to other end
        openfed_api.build_connection(address_file=address_file)

        self.openfed_api = openfed_api

        # Replace the original optimizer with pipe
        self.optimizer = optimizer

    def train(self, to: bool, data_loader=None):
        if self.rank == 0:
            if to == True:
                # We only need to update the version information when finished the training.
                # And at next time, it will automatically download the newer one.
                self.openfed_api.update_version()

            # Download or upload a model
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
