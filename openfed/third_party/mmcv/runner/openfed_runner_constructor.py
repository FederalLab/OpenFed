import os
import warnings
from typing import Dict

import openfed
import torch.distributed as dist
from mmcv.runner.builder import RUNNER_BUILDERS, RUNNERS  # type: ignore
from mmcv.runner.dist_utils import get_dist_info
from openfed.core import is_follower, is_leader, leader
from openfed.tools import build_optim

from .hooks import build_hook


@RUNNER_BUILDERS.register_module()
class OpenFedRunnerConstructor(object):
    """Default constructor for runners.
    """
    def __init__(self, runner_cfg, default_args=None):
        if not isinstance(runner_cfg, dict):
            raise TypeError('runner_cfg should be a dict',
                            f'but got {type(runner_cfg)}')

        self.role = runner_cfg.pop('role', leader)
        self.address_file = runner_cfg.pop('address_file', '')
        self.fed_optim_cfg = runner_cfg.pop('fed_optim_cfg', dict())
        self.hook_cfg_list = runner_cfg.pop('hook_cfg_list', list())

        self.runner_cfg = runner_cfg

        self.default_args: Dict = default_args  # type: ignore

    def __call__(self):
        rank, world_size = get_dist_info()

        if not self.hook_cfg_list:
            self.hook_cfg_list = [
                dict(
                    type='Aggregate',
                    count=dict(train=-1),
                    checkpoint=os.path.join(self.default_args['work_dir'],
                                            'openfed'),
                    max_version=self.default_args['max_epochs'],
                )
            ]

        if not self.fed_optim_cfg:
            self.fed_optim_cfg = dict(
                type='fedavg',
                lr=1.0,
            )
        if is_follower(self.role):
            self.fed_optim_cfg['optimizer'] = self.default_args['optimizer']
        self.fed_optim_cfg['role'] = self.role

        optimizer, aggregator = build_optim(
            self.fed_optim_cfg.pop('type'),
            self.default_args['model'].parameters(), **self.fed_optim_cfg)
        if rank == 0:
            world = openfed.core.World(role=self.role, mtt=50)
            openfed_api = openfed.API(
                world,
                self.default_args['model'].state_dict(keep_vars=True),
                optimizer,
                aggregator,
            )
            if is_leader(self.role):
                with openfed_api:
                    aggregate_in = False
                    for cfg in self.hook_cfg_list:
                        if cfg['type'] == 'Aggregate':
                            aggregate_in = True
                        build_hook(cfg)
                    if not aggregate_in:
                        warnings.warn(
                            "`Aggregate` step function is not registerred.")
            openfed_api.build_connection(address_file=self.address_file)

            if is_leader(self.role):
                # Go into leader loop backend
                openfed_api.run()
                print(">>> Finished.")
                openfed_api.finish(auto_exit=True)
                
        self.default_args['optimizer'] = optimizer

        # build existing runner
        runner = RUNNERS.build(self.runner_cfg, default_args=self.default_args)

        # Define new train function
        runner_train_func = runner.train

        def train(self, data_loader, **kwargs):
            if rank == 0:
                with openfed_api:
                    # Download a model from leader
                    openfed_api.transfer(to=False)

            # broadcast the model to other rank
            if world_size > 1:
                for p in self.model.parameters():
                    dist.broadcast(p, src=0)

            # Accumulate gradient
            self.optimizer.acg(self.model, data_loader)

            output = runner_train_func(self, data_loader, **kwargs)

            self.optimizer.round()

            instances = len(data_loader.dataset)
            openfed_api.pipe_task_info.instances = instances  # type: ignore

            # upload
            if rank == 0:
                with openfed_api:
                    # Upload the trained model to leader
                    openfed_api.update_version()

                    openfed_api.transfer(to=True)
            return output

        runner.train = train

        return runner
