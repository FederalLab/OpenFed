import time
from typing import Dict, List, Union, overload
from loguru import logger

from torch import Tensor
from torch.optim import Optimizer

import openfed

from .aggregate import Aggregator
from .common import (Address, Hook, Peeper, SafeTread, default_address)
from .federated import Destroy, Maintainer, Reign, World, openfed_lock


class Backend(SafeTread, Peeper, Hook):
    aggregator: Aggregator
    optimizer: Optimizer

    # 这里面包含了用于通讯的模型参数
    # 一般情况下是model.state_dict()
    # 如果你想要同步保存其他的数据，那请手动加入到这里面。
    # 一旦传递进来，那么这里的模型参数应该始终保持id不变。
    # 也就是说，你应该设置net.state_dict(..., keep_vars=True)
    state_dict: Dict[str, Tensor]

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
    last_aggregate_time: float

    @overload
    def __init__(self):
        """如果任何参数都不给的话，那就使用默认参数，建立一个连接。
        """

    @overload
    def __init__(self,
                 world: World = None,
                 address: Union[Address, List[Address]] = None,
                 address_file: str = None):
        """仅仅只是给定了连接相关的参数，先建立连接，后期在指定其他内容。
        """

    @overload
    def __init__(self,
                 state_dict: Dict[str, Tensor],
                 aggregator: Aggregator,
                 optimizer: Optimizer,
                 world: World = None,
                 address: Union[Address, List[Address]] = None,
                 address_file: str = None):
        """
        同时给定了所需的各种参数。
        """

    def __init__(self, **kwargs):
        super().__init__()

        self.state_dict = kwargs.get('state_dict', None)
        self.aggregator = kwargs.get('aggregator', None)
        self.optimizer = kwargs.get('optimizer', None)

        world = kwargs.get('world', None)
        address = kwargs.get('address', None)
        address_file = kwargs.get('address_file', None)

        if world is None:
            world = World(king=True)
        else:
            assert world.king, "Backend must be king."

        if address is None and address_file is None:
            address = default_address
        # NOTE: hold openfed_lock before create a dynamic address loading maintainer.
        # otherwise, it may interrupte the process and cause error before you go into loop()
        openfed_lock.acquire()

        self.maintainer = Maintainer(
            world, address=address, address_file=address_file)

        self.stopped = False

        self.version = 0
        self.reign = None

        self.received_numbers = 0
        self.last_aggregate_time = time.time()

    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        self.state_dict = state_dict

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def set_aggregator(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def safe_run(self):
        """
        如果你希望程序进入后台运行，那么请使用start()。
        如果你希望程序在前台运行，那么请直接调用run()函数。
        """
        # release openfed_lock here.
        openfed_lock.release()
        while not self.stopped:
            with self.maintainer.maintainer_lock:
                self.step_at_new_episode()
                rg = Reign.reign_generator()
                for reign in rg:
                    if not self.stopped and reign is not None:
                        self.reign = reign
                        self.step_at_first()
                        if reign.is_zombine:
                            self.step_at_zombine()
                        elif reign.is_offline:
                            # Destroy process
                            if self.step_before_destroy():
                                self.step_after_destroy(
                                    Destroy.destroy_reign(reign))
                            else:
                                self.step_at_failed()
                        elif reign.is_pushing:
                            # 表示客户端要上传数据PUSH，那么我们要download数据
                            if self.step_before_download():
                                self.step_after_download(reign.download())
                            else:
                                self.step_at_failed()
                        elif reign.is_pulling:
                            # 首先进行一些判断，来确定是否响应这个请求
                            if self.step_before_upload():
                                # 表示客户端请求下载一组数据PULL，那我们要upload数据来满足他
                                self.step_after_upload(reign.upload())
                            else:
                                self.step_at_failed()
                        else:
                            # raise Exception("Invalid state")
                            # 因为这个状态的判断过程是通过实时读取的数据进行的，所以很有可能刚做完
                            # pulling的判断，对方就恰好把状态设置成了pull，从而导致进入了valid state
                            # 但是这个并没有任何问题，不算是bug
                            # 另一种是客户端下线了，导致状态读取失败。但是也不会造成更多影响。
                            # 所以这里直接skip就好
                            ...
                    # 常规的状态检查和更新
                    self.step_at_last()
                else:
                    del rg

            # 留出时间给后台去争夺maintainer lock
            time.sleep(openfed.SLEEP_SHORT_TIME)

        self.finish()
        return "Backend exited."

    def step_at_new_episode(self):
        pass

    def step_at_first(self):
        pass

    def step_at_zombine(self):
        pass

    def step_before_destroy(self) -> bool:
        return True

    def step_after_destroy(self, state=...):
        pass

    def step_before_download(self) -> bool:
        return True

    def step_after_download(self, state=...):
        assert self.aggregator is not None, "Set aggregator first"

        # 从底层获取接收到的数据
        packages = self.reign.tensor_indexed_packages
        task_info = self.reign.task_info

        self.aggregator.step(packages, task_info)

        self.received_numbers += 1

        logger.info(f"Receive model @{self.received_numbers}")

    def step_before_upload(self) -> bool:
        """当客户端要求返回一个新的模型时，我们可能会面临不同的情况，这个申请可能无法满足。
        当无法满足的时候，返回False，否则返回True。
        这个函数中，应该包含了对package等需要传送的相关数据的设置。
        """
        assert self.optimizer is not None, "optimizer is not specified"
        assert self.aggregator is not None, "aggregator is not specified"
        assert self.state_dict is not None, "state dict is not specified"

        # 先删除之前的数据
        self.reign.reset()

        # 指定要打包的数据
        self.reign.set_state_dict(self.state_dict)

        # 打包数据
        self.reign.pack_state(self.aggregator)
        self.reign.pack_state(self.optimizer)

        # 准备发送
        return True

    def step_after_upload(self, state=...):
        pass

    def step_at_last(self):
        """用于更新内部状态
        """

        if self.received_numbers == 50:
            # 开始聚合
            task_info = self.aggregator.aggregate()

            # 更新optimizer状态
            self.aggregator.unpack_state(self.optimizer)

            # 更新梯度
            self.optimizer.step()

            # 重置状态
            self.aggregator.zero_grad()
            self.received_numbers = 0
            self.last_aggregate_time = time.time()

            self.manual_stop()
        else:
            # 暂时啥也不做
            ...

    def step_at_failed(self):
        pass

    def __repr__(self):
        return "Backend"

    def finish(self):
        # 强制杀死所有的进程，并且退出进程
        Destroy.destroy_all_in_all_world()

        self.maintainer.manual_stop()
