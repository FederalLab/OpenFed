import time
from collections import OrderedDict
from threading import Lock
from typing import Dict, List, TypeVar, Union

import openfed

from ..common import (Address, SafeTread, log_debug_info, log_error_info,
                      log_verbose_info)
from .core import FederatedWorld, ProcessGroup, Store, World, register
from .deliver import Delivery
from .inform import Informer


class Joint(SafeTread):
    """封装成一个线程，自动执行，完成建立后自动销毁。确保不会阻塞主程序的运行。
    当调用这个函数时，如果是客户端，那会阻塞，直到完成连接。
    如果是服务器端，则会进入后台运行。
    因为客户端必须要保证连接上以后，才可以进行下一步操作，而服务器端还需要管理其他连接。

    这是一个temperal的线程，不会常驻后台，故不需要监控openfed.ALIVE的状态。
    """

    build_success: bool

    def __init__(self, address: Address, world: World):
        super().__init__()

        if address.rank == -1:
            if address.world_size == 2:
                # 自动设置rank
                address.rank = 1 if world.is_queen() else 0
            else:
                msg = "Please specify the correct rank when world size is not 2"
                log_error_info(msg)
                raise RuntimeError(msg)

        self.address = address
        self.build_success = False

        self.world = world

        if self.world.is_king():
            # 如果是服务器，那就使用start函数进入后台运行
            self.start()
        else:
            # 如果是客户端，那直接调用run函数，确保整个连接顺利完成。
            self.start()
            self.join()

    def safe_run(self):
        """Build a new federated world.
        一般情况下，只有两个成员（服务器和客户端）。
        特殊情况下，会有N个成员（N>2）共享这个通讯世界，以减少创建连接带来的过多开销，
        同时也简化代码逻辑。这种情况下，服务器永远是rank=0，但是client则要根据实际情况指定。
        当N>2的时候，我们依然会建立P2P的通讯，只不过这些通讯都是在同一组通讯世界下。

        在一个程序的运行过程中，你可以反复的调用该函数，建立许多不同的连接。这使得你可以自由的控制
        客户端的增加，而不需要在一开始就指定所有的客户端。
        """
        log_verbose_info(f"Connect to \n{str(self.address)}")

        # create a federated world and auto register it
        fed_world = FederatedWorld()

        # build the connection between the federated world
        try:
            fed_world.init_process_group(**self.address.as_dict)
        except Exception as e:
            del fed_world
            return f"Timeout while building connection to {repr(self.address)}"

        # register the world
        with self.world.joint_lock:
            register.register_federated_world(fed_world, self.world)

        # 无论在客户端还是服务器端，这里的rank都是指定0，也就是服务器端。
        sub_pg_list = fed_world.build_point2point_group(rank=0)

        # bound pg with the federated world
        for sub_pg in sub_pg_list:
            reign = Reign(fed_world.get_store(
                sub_pg), sub_pg, fed_world, self.world)
            with self.world.joint_lock:
                self.world._pg_mapping[sub_pg] = reign

        self.build_success = True
        log_verbose_info(f"Conneted to {repr(self.address)}")

    def __repr__(self):
        return "Joint"


_M = TypeVar("_M", bound="Maintainer")
# 这个锁，非常重要！
# 每当你创建一个新的maintainer时候，都会生成一个这种锁。
# 这个锁的作用是为了防止后台建立连接的时候，造成所有线程阻塞
# 这会导致正在处理的其他线程发生错误！
# 如果你不需要动态加入新的联邦世界的话，那不需要考虑这个问题。
# 因为你一开始就指定了所有的节点，并且直到所有的节点都加入进来后
# 才会继续运行接下来的程序。
_maintainer_lock_dict: Dict[_M, Lock] = OrderedDict()

# 这是一把很特殊的锁，当你使用这个锁的时候，可以保证你的主线程在任何情况下都不会被
# 后台创建连接的操作打断。除非你主动交出这把锁，否则不再会有任何新的。
# 连接被创建。但是除此之外的openfed的其他功能不会受到任何影响。
openfed_lock = Lock()


class Maintainer(SafeTread):
    """负责在后台自动建立连接。
    在服务器端，该片段代码会自动启动一个新的进程，进行连接控制。
    在客户端，该程序需要由手动调用建立连接：因为客户端的程序，往往只需要建立一次连接就够，并不需要不断地监听信号。
    """
    # 用来记录正在等待的连接，如果需要添加新的连接，请加入到这个列表中。
    pending_queue: List[Address]
    # 用来记录已完成的连接。
    finished_queue: List[Address]

    maintainer_lock: Lock
    # 用来控制所有不同连接的联邦世界。
    world: World

    def __init__(self,
                 world: World,
                 address: Union[Address, List[Address]] = None,
                 address_file: str = None):
        """
            在客户端，一次只允许指定一个地址，如果多余一个地址，则会报错。
        """
        super().__init__()

        self.maintainer_lock = Lock()
        _maintainer_lock_dict[self] = self.maintainer_lock

        self.world = world

        self.pending_queue = list()
        self.finished_queue = list()

        self.address_file = address_file

        if address is not None:
            if not isinstance(address, (list, tuple)):
                address = [address]
        else:
            address = []

        if self.world.is_king():
            self.pending_queue.extend(address)
            self.pending_queue.extend(self.read_address_from_file())

            self.start()
            if not openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                # 如果不是动态加载的话，则会阻塞进程，直到所有的连接都正确建立
                self.join()
        else:
            # 如果是客户端，并且给定连接地址的话，则直接连接
            self.pending_queue.extend(address)
            self.pending_queue.extend(self.read_address_from_file())

            if len(self.pending_queue) > 1:
                msg = "Too many fed addr are specified. Only allowed 1."
                log_error_info(msg)
                raise RuntimeError(msg)
            elif len(self.pending_queue) == 1:
                address = self.pending_queue.pop()
                Joint(address, self.world)
                self.finished_queue.append(address)
            else:
                log_verbose_info(
                    "Waiting for a valid address to build a connection.")

    def read_address_from_file(self) -> List[Address]:
        if self.address_file is None:
            return []
        address_list = Address.read_address_from_file(self.address_file)
        valid_address_list = []
        # 判断address_list中的值，是否已经存在，如果已经存在，则不再重复创建
        for address in self.pending_queue + self.finished_queue:
            for address_ in address_list:
                if address.init_method == address_.init_method:
                    # 已经存在
                    ...
                else:
                    valid_address_list.append(address_)

        return valid_address_list

    def safe_run(self):
        while not self.stopped and self.world.ALIVE:
            # 读取文件，自动添加
            new_addr = self.read_address_from_file()
            self.pending_queue.extend(new_addr)

            def acquire_all():
                # 拿到所有的锁
                for maintainer_lock in _maintainer_lock_dict.values():
                    maintainer_lock.acquire()
                openfed_lock.acquire()

            def release_all():
                # 释放所有的锁
                for maintainer_lock in _maintainer_lock_dict.values():
                    maintainer_lock.release()
                openfed_lock.release()

            build_failed = []
            if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                for address in self.pending_queue:
                    acquire_all()
                    joint = Joint(address, self.world)
                    joint.join()
                    release_all()
                    if joint.build_success:
                        self.finished_queue.append(address)
                        # 等待一小段时间再去获取锁，为了把机会留给其他人
                        time.sleep(openfed.SLEEP_SHORT_TIME)
                    else:
                        build_failed.append(address)
                        # 等待一段时间再去获取锁，为了把机会留给其他人
                        time.sleep(openfed.SLEEP_LONG_TIME)
            else:
                joint_address_mapping = []
                for address in self.pending_queue:
                    joint = Joint(address, self.world)
                    joint_address_mapping.append([joint, address])

                for joint, address in joint_address_mapping:
                    joint.join()
                    if joint.build_success:
                        self.finished_queue.append(address)
                    else:
                        build_failed.append(address)
            self.pending_queue = build_failed

            if len(self.pending_queue) == 0:
                if openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading:
                    # 如果没有排队等待，那就睡眠长一些！减少CPU占用
                    time.sleep(openfed.SLEEP_LONG_TIME)
                else:
                    # 退出for循环，表示已经完成所有的连接任务
                    break
            else:
                time.sleep(openfed.SLEEP_LONG_TIME)

    def kill_world(self):
        # world一旦kill，所有相关的后台程序都会立刻结束
        self.world.killed()

    def manual_stop(self, kill_world: bool = True):
        if kill_world:
            self.kill_world()
        super().manual_stop()

    def manual_joint(self, address: Address):
        """如果是客户端，则直接连接，会阻塞操作。如果是服务器，则加入队列，让后台自动连接。
        """
        if not openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading and self.world.is_king():
            raise RuntimeError("Dynamic loading is not allowed!")

        log_debug_info(f"Add a new address {repr(address)} manually.")
        if self.world.is_king():
            self.pending_queue.append(address)
        else:
            Joint(address, self.world)

    def __repr__(self):
        return "Maintainer"

    def __del__(self):
        # 删除锁！！！
        del _maintainer_lock_dict[self]
        super().__del__()


class Destroy(object):
    """销毁客户端。
    """
    @classmethod
    def destroy(cls, pg: ProcessGroup, world: World = None):
        """如果不指定world，则使用默认组。也就是在__federated_world__中的第一个。
        """
        if world is None:
            world = register.default_world

        # 如果删除的是当前的pg，不要忘了重置
        if pg == world._current_pg:
            world._current_pg = world._NULL_GP

        # 获取对应的federated_world
        reign = world._pg_mapping[pg]

        # 将reign状态设置成OFFINE
        reign.offline()

        # 将键值对从全局字典中移除
        del world._pg_mapping[pg]
        federated_world = reign.federated_world
        # 删除PG
        federated_world.destroy_process_group(pg)

        # 判断是否需要直接删除fed world
        if not federated_world.is_initialized() or federated_world._group_count == 1:
            # 说明删除pg后，这个fed world没有任何pg存在
            # 因此可以直接删除

            # 说明删除之后，还剩下一个全局gp
            # 就说明这是一个共享的fed world。
            # 剩下全局pg的时候，就说明其他的客户端都退出，所以这里可以直接删除
            register.deleted_federated_world(federated_world)
        else:
            # 说明这个世界中还有其他客户端，因此，暂时不能断开全局gp。
            ...

    @classmethod
    def destroy_current(cls, world: World = None):
        if world is None:
            world = register.default_world
        cls.destroy(world._current_pg, world)

    @classmethod
    def destroy_all_in_a_world(cls, world: World = None):
        if world is None:
            world = register.default_world
        for pg, _ in world:
            if pg is not None:
                cls.destroy(pg, world)

    @classmethod
    def destroy_all_in_all_world(cls):
        """如果你想结束联邦学习，那调用这个函数
        """
        for _, world in register:
            if world is not None:
                cls.destroy_all_in_a_world(world)


class Reign(Informer, Delivery):
    """将进行‘沟通相关’的对象全部集合起来，并且提供进一步的封装，交付上层使用。
    """
    store: Store
    pg: ProcessGroup
    world: World
    federated_world: FederatedWorld

    # version 是用来标明当前端数据的版本号的。
    version: int

    def __init__(self,
                 store: Store,
                 pg: ProcessGroup,
                 federated_world: FederatedWorld,
                 world: World,
                 ):
        self.pg = pg
        self.store = store
        self.federated_world = federated_world
        self.world = world

        Informer.__init__(self)
        Delivery.__init__(self)

        self.version = 0
        self.set("version", self.version)

    def upload(self) -> bool:
        """将package中的数据传送到另一方并且处理相关逻辑。
        你在任何情况下，都应该通过这种方式，upload数据，而不是直接使用package.push操作。

        Returns: 返回一个bool，来表示操作是否成功。
        """

        # 记住一个原则：设置的永远都是自己的状态，读取的永远都是对方的状态
        if self.world.is_queen():
            # 1. 写入自身版本号
            self.set('version', self.version)
            # 2. 设置STATUE状态为PUSH，告知另一端自己等待上传数据
            self.pushing()
            # 3. 进入阻塞，等待数据上传
            tic = time.time()
            while not self.is_pulling:  # 检查对方是否进入了pulling状态
                toc = time.time()
                if toc-tic > openfed.SLEEP_VERY_LONG_TIME or self.is_offline:
                    return False
                time.sleep(openfed.SLEEP_SHORT_TIME)

            self.push()
            # 4. 设置自己的状态为ZOMBINE
            self.zombine()

            return True
        else:
            # 1. 写入服务器端的版本号
            self.set('version', self.version)
            # 2. 设置状态为完成ZOMBINE，先设置状态，在推送数据。
            # 防止通信延迟造成的逻辑错误。
            # 比如，当客户端上传完模型，现在需要再次下载一个新的模型时，
            # 客户端可能会立刻将state设置为PULL
            # 而此时服务器端才接收完模型，其要将ZOMBINE设置到状态里。
            # 但是由于延时，可能导致其在PULL生效之后，才写入的ZOMBINE
            # 那么，这时候该客户端将进入无限期的等待中而不会得到服务器的响应。
            self.pushing()
            # 3. 发送数据
            self.push()
            # 4. 设置自己的状态为ZOMBINE
            self.zombine()
            return True

    def download(self) -> bool:
        """从另一方接收package数据。
        你在任何情况下，都应该通过这种方式，download数据，而不是直接使用package.pull操作。

        Returns: 返回一个操作数来表示操作是否成功。

        客户端会等待一段时间，服务器则直接返回falied。
        """
        if self.world.is_queen():
            # 1. 写入自身版本号
            self.set('version', self.version)
            # 2. 设置STATUE状态为PULL，告知另一端自己等待下载数据
            self.pulling()
            # 3. 进入阻塞，等待数据上传
            tic = time.time()
            while not self.is_pushing:  # 检查对方是否进入了pushing状态
                toc = time.time()
                if toc-tic > openfed.SLEEP_VERY_LONG_TIME or self.is_offline:
                    return False
                time.sleep(openfed.SLEEP_SHORT_TIME)

            self.pull()
            # 4. 设置自己的状态为ZOMBINE
            self.zombine()
            return True
        else:
            # 1. 写入服务器端的版本号
            self.set('version', self.version)
            # 2. 设置状态为完成ZOMBINE。先设置状态，再推送数据。
            self.pulling()
            # 3. 发送数据
            self.pull()
            # 4. 设置自己的状态为ZOMBINE
            self.zombine()
            return True

    def destroy(self):
        """退出联邦学习。
        """
        # 销毁当前进程
        Destroy.destroy(self.pg, self.world)

    def __repr__(self) -> str:
        string = "Reign\n"
        string += f"Version: {self.version}\n"
        string += f"Status: {self._get_state()}\n"
        return string


def reign_generator() -> Reign:
    # 以下的这套逻辑主要是为了应对register和world中的reign动态的修改
    for _, world in register:
        if world is None and not world.ALIVE:
            break
        for pg, reign in world:
            if reign is None and not world.ALIVE:
                break
            yield reign
            world._current_pg = pg
    else:
        # 当列表为空的时候，yield一个空的GP
        # 否则无法进入for循环的话，将无法形成一个Generator
        yield None


def default_reign() -> Reign:
    """
    返回唯一的一个reign。
    当你拥有唯一的一个pg的时候，调用这个函数，会返回由它组成的reign。
    当你作为客户端的时候，调用这个函数可以方便的得到reign，简化代码。
    """
    # 确保整个环境已经初始化
    if len(register) == 0:
        raise RuntimeError("Please build a federated world first!")
    assert len(register) == 1, "More than one federated world."
    return register.default_world.default_reign
