import time
from threading import Thread
from typing import List, Union

import openfed
import openfed.utils as utils
from openfed.utils.types import STATUS, FedAddr

from .core.federated_c10d import FederatedWorld, ProcessGroup
from .monitor.monitor import Monitor
from .pack.package import Package
from .register import World, register
from .utils.safe_exited import get_head_info, safe_exited


class Joint(Thread):
    """封装成一个线程，自动执行，完成建立后自动销毁。确保不会阻塞主程序的运行。
    当调用这个函数时，如果是客户端，那会阻塞，直到完成连接。
    如果是服务器端，则会进入后台运行。
    因为客户端必须要保证连接上以后，才可以进行下一步操作，而服务器端还需要管理其他连接。

    这是一个temperal的线程，不会常驻后台，故不需要监控openfed.ALIVE的状态。
    """

    def __init__(self, fed_addr: FedAddr, world: World):
        super().__init__(name="OpenFed Joint Thread")

        if fed_addr.rank == -1:
            if fed_addr.world_size == 2:
                # 自动设置rank
                fed_addr.rank = 1 if world.is_queen() else 0
            else:
                raise RuntimeError(
                    "Please specify the correct rank when world size is not 2")

        if openfed.VERBOSE:
            print(utils.yellow_color("Connect..."), f"{fed_addr}")

        self.fed_addr = fed_addr

        self.world = world

        self.stopped = False

        if self.world.is_king():
            # 如果是服务器，那就使用start函数进入后台运行
            self.start()
        else:
            # 如果是客户端，那直接调用run函数，确保整个连接顺利完成。
            self.start()
            self.join()

    def run(self):
        """Build a new federated world.
        一般情况下，只有两个成员（服务器和客户端）。
        特殊情况下，会有N个成员（N>2）共享这个通讯世界，以减少创建连接带来的过多开销，
        同时也简化代码逻辑。这种情况下，服务器永远是rank=0，但是client则要根据实际情况指定。
        当N>2的时候，我们依然会建立P2P的通讯，只不过这些通讯都是在同一组通讯世界下。

        在一个程序的运行过程中，你可以反复的调用该函数，建立许多不同的连接。这使得你可以自由的控制
        客户端的增加，而不需要在一开始就指定所有的客户端。
        """
        # create a federated world and auto register it
        fed_world = FederatedWorld()

        # register the world
        with self.world.joint_lock:
            register.register_federated_world(fed_world, self.world)

        # build the connection between the federated world
        fed_world.init_process_group(**self.fed_addr.as_dict())

        # 无论在客户端还是服务器端，这里的rank都是指定0，也就是服务器端。
        sub_pg_list = fed_world.build_point2point_group(rank=0)

        # bound pg with the federated world
        for sub_pg in sub_pg_list:
            monitor = Monitor(fed_world.get_store(
                sub_pg), fed_world, self.world)
            package = Package(sub_pg, fed_world, self.world)
            with self.world.joint_lock:
                self.world._pg_mapping[sub_pg] = [package, monitor, fed_world]
        if openfed.VERBOSE:
            print(utils.green_color("Connected"), f"{self.fed_addr}")

        self.stopped = True

    def manual_stop(self):
        self.stopped = True


class Maintainer(Thread):
    """负责在后台自动建立连接。
    在服务器端，该片段代码会自动启动一个新的进程，进行连接控制。
    在客户端，该程序需要由手动调用建立连接：因为客户端的程序，往往只需要建立一次连接就够，并不需要不断地监听信号。
    """
    # 用来记录正在等待的连接，如果需要添加新的连接，请加入到这个列表中。
    pending_queue: List[FedAddr]
    # 用来记录已完成的连接。
    finished_queue: List[FedAddr]

    def __init__(self,
                 world: World,
                 fed_addr: Union[FedAddr, List[FedAddr]] = None,
                 fed_addr_file: str = None):
        """
            在客户端，一次只允许指定一个地址，如果多余一个地址，则会报错。
        """
        super().__init__()
        self.world = world

        self.stopped = False
        self.pending_queue = list()
        self.finished_queue = list()

        self.fed_addr_file = fed_addr_file

        if fed_addr is not None:
            if not isinstance(fed_addr, (list, tuple)):
                fed_addr = [fed_addr]
        else:
            fed_addr = []

        if self.world.is_king():
            self.pending_queue.extend(fed_addr)
            self.pending_queue.extend(self.read_fed_addr_from_file())

            self.start()
        else:
            # 如果是客户端，并且给定连接地址的话，则直接连接
            self.pending_queue.extend(fed_addr)
            self.pending_queue.extend(self.read_fed_addr_from_file())

            if len(self.pending_queue) > 1:
                raise RuntimeError(
                    "Too many fed addr are specified. Only allowed 1.")
            elif len(self.pending_queue) == 1:
                fed_addr = self.pending_queue.pop()
                Joint(fed_addr, self.world)
                self.finished_queue.append(fed_addr)
            else:
                if openfed.VERBOSE:
                    print(utils.red_color("Addr Missed"))

    def read_fed_addr_from_file(self) -> List[FedAddr]:
        if self.fed_addr_file is None:
            return []
        fed_addr_list = FedAddr.read_fed_addr_from_file(self.fed_addr_file)
        valid_fed_addr_list = []
        # 判断fed_addr_list中的值，是否已经存在，如果已经存在，则不再重复创建
        for fed_addr in self.pending_queue + self.finished_queue:
            for fed_addr_ in fed_addr_list:
                if fed_addr.init_method == fed_addr_.init_method:
                    # 已经存在
                    ...
                else:
                    valid_fed_addr_list.append(fed_addr_)

        return valid_fed_addr_list

    def run(self):
        while not self.stopped and self.world.ALIVE:
            # 读取文件，自动添加
            new_addr = self.read_fed_addr_from_file()
            self.pending_queue.extend(new_addr)

            for fed_addr in self.pending_queue:
                Joint(fed_addr, self.world)
            # mv pending_queue to finished_queue
            # 小心！这里不允许使用append方法，否则clear之后，数据会被同时清空。
            self.finished_queue.extend(self.pending_queue)
            self.pending_queue = []

            time.sleep(self.world.SLEEP_SHORT_TIME)
        else:
            safe_exited(get_head_info())

    def manual_stop(self):
        """Provide a function to end it manually.
        """
        self.stopped = True

    def manual_joint(self, fed_addr: FedAddr):
        """如果是客户端，则直接连接，会阻塞操作。如果是服务器，则加入队列，让后台自动连接。
        """
        if self.world.is_king():
            self.pending_queue.append(fed_addr)
        else:
            Joint(fed_addr, self.world)


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
        package, monitor, fed_world = world._pg_mapping[pg]

        # 将informer状态设置成OFFINE
        monitor.set_state(STATUS.OFFLINE)
        monitor.manual_stop()

        # 将键值对从全局字典中移除
        del world._pg_mapping[pg]

        # 删除PG
        fed_world.destroy_process_group(pg)

        # 判断是否需要直接删除fed world
        if not fed_world.is_initialized():
            # 说明删除pg后，这个fed world没有任何pg存在
            # 因此可以直接删除
            world.killed()
            register.deleted_federated_world(fed_world)
        elif fed_world._group_count == 1:
            # 说明删除之后，还剩下一个全局gp
            # 就说明这是一个共享的fed world。
            # 剩下全局pg的时候，就说明其他的客户端都退出，所以这里可以直接删除
            world.killed()
            register.deleted_federated_world(fed_world)
        else:
            # 说明这个世界中还有其他客户端，因此，暂时不能断开全局gp。
            ...

    @classmethod
    def destroy_current(cls, world: World = None):
        if world is None:
            world = register.default_world
        cls.destroy(world._current_pg, world)

    @classmethod
    def destroy_all(cls, world: World = None):
        if world is None:
            world = register.default_world
        for pg in world._pg_mapping:
            cls.destroy(pg, world)


class Reign(object):
    """将进行‘沟通相关’的对象全部集合起来，并且提供进一步的封装，交付上层使用。
    """
    pg: ProcessGroup
    world: World
    package: Package
    monitor: Monitor
    federated_world: FederatedWorld

    # version 是用来标明当前端数据的版本号的。
    version: int

    def __init__(self,
                 pg: ProcessGroup,
                 world: World,
                 package: Package,
                 monitor: Monitor,
                 federated_world: FederatedWorld):
        self.pg = pg
        self.world = world
        self.package = package
        self.monitor = monitor
        self.federated_world = federated_world
        self.version = 0

        self.monitor.set("version", self.version)

    def upload(self):
        """将package中的数据传送到另一方并且处理相关逻辑。
        你在任何情况下，都应该通过这种方式，upload数据，而不是直接使用package.push操作。
        """
        if self.world.is_queen():
            # 1. 写入自身版本号
            self.monitor.set('version', self.version)
            # 2. 设置STATUE状态为PUSH，告知另一端自己等待上传数据
            self.monitor.set_state(STATUS.PUSH)
            # 3. 进入阻塞，等待数据上传
            self.package.push()
            # 4. 设置自己的状态为ZOMBINE
            self.monitor.set_state(STATUS.ZOMBINE)
        else:
            # 1. 写入服务器端的版本号
            self.monitor.set('version', self.version)
            # 2. 设置状态为完成ZOMBINE，先设置状态，在推送数据。
            # 防止通信延迟造成的逻辑错误。
            # 比如，当客户端上传完模型，现在需要再次下载一个新的模型时，
            # 客户端可能会立刻将state设置为PULL
            # 而此时服务器端才接收完模型，其要将ZOMBINE设置到状态里。
            # 但是由于延时，可能导致其在PULL生效之后，才写入的ZOMBINE
            # 那么，这时候该客户端将进入无限期的等待中而不会得到服务器的响应。
            self.monitor.set_state(STATUS.PULL)
            # 3. 发送数据
            self.package.push()
            # 4. 设置自己的状态为ZOMBINE
            self.monitor.set_state(STATUS.ZOMBINE)

    def download(self):
        """从另一方接收package数据。
        你在任何情况下，都应该通过这种方式，download数据，而不是直接使用package.pull操作。
        """
        if self.world.is_queen():
            # 1. 写入自身版本号
            self.monitor.set('version', self.version)
            # 2. 设置STATUE状态为PULL，告知另一端自己等待下载数据
            self.monitor.set_state(STATUS.PULL)
            # 3. 进入阻塞，等待数据上传
            self.package.pull()
            # 4. 设置自己的状态为ZOMBINE
            self.monitor.set_state(STATUS.ZOMBINE)
        else:
            # 1. 写入服务器端的版本号
            self.monitor.set('version', self.version)
            # 2. 设置状态为完成ZOMBINE。先设置状态，再推送数据。
            self.monitor.set_state(STATUS.PUSH)
            # 3. 发送数据
            self.package.pull()
            # 4. 设置自己的状态为ZOMBINE
            self.monitor.set_state(STATUS.ZOMBINE)

    def destroy(self):
        """退出联邦学习。
        """
        # 销毁当前进程
        Destroy.destroy(self.pg, self.world)


def process_generator() -> Reign:
    """生成器，不断的遍历整个pg数组，并且返回一个pg。
    注意：返回的pg可能是无效的。
        当不存在pg时，会返回一个None。
        由于yield是提前准备数据，那么pg可能被删除。
    故，需要判断！
    """
    while len(register):
        for fed_world, world in register:
            for pg in world._pg_mapping:
                if not world.ALIVE:
                    # 只有在确保openfed存活的状态下，才维持这个生成器
                    # 否则的话，自动结束这个线程。
                    break
                # 注意以下代码的运行逻辑
                # 在生成器语法中，程序会进入后台执行。
                # 当执行到yield语句时，程序将会阻塞，等待数据被取走。
                # 当数据被取走后，程序才会继续向下执行。
                # 因此，在这里，我们应该先等待pg被取走
                # 然后再去将_current_pg更新成被取走的pg
                # 如果先更新_current_pg的话，会导致实际的_current_pg指向发生错误
                yield Reign(pg, world, *world._pg_mapping[pg])
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
    for fed_world, world in register:
        for pg in world._pg_mapping:
            world._current_pg = pg
            return Reign(pg, world, *world._pg_mapping[pg])
