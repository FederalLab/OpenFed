import time
from threading import Thread
from typing import Dict, List, Union

from torch import Tensor
from torch.optim import Optimizer

import openfed.federated as federated

from .aggregate import Aggregator
from .federated.federated import Maintainer, Reign, World
from .utils.types import STATUS, FedAddr


class Backend(Thread):
    aggregator: Aggregator
    optimizer: Optimizer

    # 这里面包含了用于通讯的模型参数
    # 一般情况下是model.state_dict()
    # 如果你想要同步保存其他的数据，那请手动加入到这里面。
    # 一旦传递进来，那么这里的模型参数应该始终保持id不变。
    # 也就是说，你应该设置net.state_dict(..., keep_vars=True)
    state_dict: Dict[str, Tensor]

    stopped: bool

    # 一个maintiner用于处理连接
    maintiner: Maintainer

    # 当前正在处理的对象
    reign: Reign

    # 用来标志当前模型的版本信息
    # 每更新一次全局模型，这个版本号递增
    version: int

    # 记录收集到的模型的数量，可以根据这个来判断是否进行aggregate
    received_numbers: int

    # 记录上一次模型更新的时间，可以根据这个来判断是否超时强制更新
    last_time: float

    def __init__(self,
                 state_dict: Dict[str, Tensor],
                 aggregator: Aggregator,
                 optimizer: Optimizer,
                 world: World = None,
                 fed_addr: Union[FedAddr, List[FedAddr]] = None,
                 fed_addr_file: str = None
                 ):
        self.state_dict = state_dict
        self.aggregator = aggregator
        self.optimizer = optimizer
        self.stopped = False

        self.version = 0
        self.reign = None

        self.received_numbers = 0
        self.last_time = time.time()

        if world is None:
            world = World()
            world.set_king()
        else:
            assert world.is_king(), "Backend must be king."
        self.maintainer = Maintainer(
            world, fed_addr=fed_addr, fed_addr_file=fed_addr_file)

    def run(self):
        """
        如果你希望程序进入后台运行，那么请使用start()。
        如果你希望程序在前台运行，那么请直接调用run()函数。
        """
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
        # 从底层获取接收到的数据
        packages = self.reign.package.tensor_indexed_packages
        task_info = self.reign.monitor.get_task_info()

        self.aggregator.step(packages, task_info)

        self.received_numbers += 1

    def before_send_a_new_model(self) -> bool:
        """当客户端要求返回一个新的模型时，我们可能会面临不同的情况，这个申请可能无法满足。
        当无法满足的时候，返回False，否则返回True。
        这个函数中，应该包含了对package等需要传送的相关数据的设置。
        """
        # 指定要打包的数据。
        self.reign.package.state_dict_map(self.state_dict)

        # 打包数据
        self.reign.package.pack_state(self.aggregator)
        self.reign.package.pack_state(self.optimizer)

        # 准备发送
        return True

    def update(self):
        """用于更新内部状态
        """

        if self.received_numbers == 5:
            # 开始聚合
            task_info = self.aggregator.aggregate()

            print(task_info)

            # 更新optimizer状态
            self.aggregator.unpack_state(self.optimizer)

            # 更新梯度
            self.optimizer.step()

            # 重置状态
            self.aggregator.zero_grad()
            self.received_numbers = 0
            self.last_time = time.time()
        else:
            # 暂时啥也不做
            ...

    def manual_stop(self):
        self.stopped = True
