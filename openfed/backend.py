from threading import Thread

from torch.optim import Optimizer

import openfed.federated as federated
from openfed.federated.federated import FederatedTuple
from aggregate import Aggregator
from .utils.types import STATUS


class Backend(Thread):
    aggregator: Aggregator
    optimizer: Optimizer
    stopped: bool
    federated_tuple: FederatedTuple

    def __init__(self, aggregator: Aggregator, optimizer: Optimizer):
        self.aggregator = aggregator
        self.optimizer = optimizer
        self.stopped = False

    def run(self):
        while not self.stopped:
            for federated_tuple in federated.process_generator():
                self.federated_tuple = federated_tuple

                pg, world, package, monitor, federated_world = federated_tuple
                if world.is_valid_process_group(pg):
                    state = monitor.informer.get_state()

                    if state == STATUS.ZOMBINE:
                        # Do nothing, skip
                        ...
                    elif state == STATUS.OFFLINE:
                        # Destory process
                        federated.Destroy.destroy(pg, world)
                    elif state == STATUS.PUSH:
                        # Push process
                        package.pull()
                        # Reset Flag
                        monitor.informer.set_state(STATUS.ZOMBINE)

                        tensor_indexed_packages = package.tensor_indexed_packages
                        task_info = monitor.informer.get_task_info()

                        self.aggregator.step(
                            tensor_indexed_packages, task_info)
                    elif state == STATUS.PULL:
                        # Pull process
                        package.push()
                        # Reset Flag
                        monitor.informer.set_state(STATUS.ZOMBINE)
                    else:
                        raise Exception("Invalid state")
                self.update()

    def update(self):
        """用于更新内部状态
        """

    def manual_stop(self):
        self.stopped = True
