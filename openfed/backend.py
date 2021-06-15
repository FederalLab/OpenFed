from threading import Thread

from torch.optim import Optimizer

import openfed.federated as federated
from openfed.federated.federated import Reign
from aggregate import Aggregator
from .utils.types import STATUS


class Backend(Thread):
    aggregator: Aggregator
    optimizer: Optimizer
    stopped: bool

    # 当前正在处理的对象
    reign: Reign

    # 用来标志当前模型的版本信息
    # 每更新一次全局模型，这个版本号递增
    version: int

    def __init__(self, aggregator: Aggregator, optimizer: Optimizer):
        self.aggregator = aggregator
        self.optimizer = optimizer
        self.stopped = False

        self.version = 0
        self.reign = None

    def run(self):
        while not self.stopped:
            for reign in federated.process_generator():
                self.reign = reign

                if reign is not None:
                    state = reign.monitor.get_state()

                    if state == STATUS.ZOMBINE:
                        # Do nothing, skip
                        ...
                    elif state == STATUS.OFFLINE:
                        # Destory process
                        reign.destroy()
                    elif state == STATUS.PUSH:
                        # 表示客户端要上传数据PUSH，那么我们要download数据
                        reign.download()
                        self.after_received_a_new_model()
                    elif state == STATUS.PULL:
                        # 首先进行一些判断，来确定是否响应这个请求
                        if self.before_send_a_new_model():
                            # 表示客户端请求下载一组数据PULL，那我们要upload数据来满足他
                            reign.upload()
                    else:
                        raise Exception("Invalid state")
                # 常规的状态检查和更新
                self.update()

    def after_received_a_new_model(self):
        # 收集到客户端模型后，进行进一步的处理，例如aggregate等等
        tensor_indexed_packages = package.tensor_indexed_packages
        task_info = monitor.get_task_info()

        self.aggregator.step(
            tensor_indexed_packages, task_info)

    def before_send_a_new_model(self) -> bool:
        """当客户端要求返回一个新的模型时，我们可能会面临不同的情况，这个申请可能无法满足。
        当无法满足的时候，返回False，否则返回True。
        这个函数中，应该包含了对package等需要传送的相关数据的设置。
        """

    def update(self):
        """用于更新内部状态
        """

    def manual_stop(self):
        self.stopped = True
