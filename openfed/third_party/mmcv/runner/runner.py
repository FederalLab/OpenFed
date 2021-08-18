from mmcv.runner.builder import RUNNERS, build_runner
from openfed.common.gluer import glue
from typing import Any

from .builder import build_openfed


def openfed_runner(model,
                   batch_processor=None,
                   optimizer=None,
                   work_dir=None,
                   logger=None,
                   meta=None,
                   max_iters=None,
                   max_epochs=None,
                   runner_cfg=None,
                   openfed_cfg=None):
    # Build Runner
    runner = build_runner(runner_cfg,
                          default_args=dict(
                              model=model,
                              batch_processor=batch_processor,
                              optimizer=optimizer,
                              work_dir=work_dir,
                              logger=logger,
                              meta=meta,
                              max_iters=max_iters,
                              max_epochs=max_epochs,
                          ))

    # Build OpenFed on rank 0 only.
    openfed = build_openfed(openfed_cfg,
                            default_args=dict(
                                model=model,
                                optimizer=optimizer,
                                rank=runner.rank,
                            ))

    if openfed.openfed_api.leader:
        # if leader, go to backthread
        openfed.openfed_api.run()
        openfed.openfed_api.finish(auto_exit=True)

    def train(func_a, func_b):
        def _train(self, *args, **kwargs):
            with self.openfed_api:
                data_loader = args[0]

                # download a model
                func_a(self, to=False, data_loader=data_loader)

                output = func_b(self, *args, **kwargs)

                self.optimizer.round()

                instances = len(data_loader.dataset)
                # collect training instances
                self.openfed_api.delivery_task_info.instances = instances

                # upload trained model
                func_a(self, to=True)  # upload a model

            return output

        return _train

    TypeA = type(openfed)
    TypeB = type(runner)
    openfed_runner = glue(
        openfed,
        runner,
        extra_func=dict(
            train=train(getattr(TypeA, 'train'), getattr(TypeB, 'train'))))

    return openfed_runner


@RUNNERS.register_module(name='OpenFedRunner')
class OpenFedRunner(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.openfed_runner = openfed_runner(*args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        if name == 'openfed_runner':
            return super().__getattribute__(name)
        return getattr(self.openfed_runner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'openfed_runner':
            return super().__setattr__(name, value)
        return setattr(self.openfed_runner, name, value)
