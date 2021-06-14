from collections import OrderedDict
from typing import Any, Dict, List, overload

import openfed.utils.types as types

from .core.federated_c10d import FederatedWorld, ProcessGroup
from .monitor.monitor import Monitor
from .pack.package import Package


class World(object):
    """World里面所有的状态，都只是给本地程序使用的。
    比如ALIVE状态，标志的是本地的这个程序是否还在运行。
    这个和informer里面的状态的含义是不一样的。
    informer里面的状态指的是联邦学习过程中的相关状态。因此不要混淆。
    """
    # 标志当前openfed world是否还在运行
    # 当ALIVE=False，程序的相关进程会自从退出
    # 防止程序运行时错误
    # 当你想要退出一个openfed world时，你应该设置这个flag
    ALIVE: bool

    def killed(self):
        # 退出
        self.ALIVE = False

    # 如果DEBUG=True，那相关的程序会输出部分调试信心
    DEBUG: bool

    def debug(self):
        self.DEBUG = True

    # 如果VERBOSE=True, 相关程序会输出一些日志
    VERBOSE: bool

    def verbose(self):
        self.VERBOSE = True

    # 给APPROVED指定不同等级的权限信息
    APPROVED: types.APPROVED

    def set_approved(self, approved: types.APPROVED):
        self.APPROVED = approved

    # ROLE用来明确当前世界中自己的身份
    # 默认所有的KING是rank=0，所有的QUEEN是rank=1
    ROLE: types.ROLE

    def set_king(self):
        self.ROLE = types.ROLE.KING

    def set_queen(self):
        self.ROLE = types.ROLE.QUEEN

    def is_king(self):
        return self.ROLE == types.ROLE.KING

    def is_queen(self):
        return self.ROLE == types.ROLE.QUEEN

    @overload
    def set_openfed_state(self, state: types.APPROVED):
        """自动设置一个权限，用来控制上传的系统信息。
        """

    @overload
    def set_openfed_state(self, state: types.ROLE):
        """设置当前进程的身份。
        """

    @overload
    def set_openfed_state(self, state):
        """根据state的类型，类设置正确的变量。
        """
        if isinstance(state, types.ROLE):
            self.ROLE = state
        elif isinstance(state, types.APPROVED):
            self.APPROVED = state
        else:
            raise NotImplementedError

    # 我们遍历的是pg，而不是federated_world。
    __pg_mapping: Dict[ProcessGroup, List[Package, Monitor, FederatedWorld]]

    # 记录当前上层正在处理的pg是哪一个
    __NULL_GP: Any
    __current_pg: ProcessGroup

    # 我们并不希望这个参数被轻易修改，所以将它定义在这里，而不是CONSTANT里面。
    SLEEPTIME: float  # seconds

    def __init__(self, ):
        self.ALIVE = True
        self.DEBUG = False
        self.VERBOSE = False
        self.APPROVED = types.APPROVED.ALL
        self.ROLE = types.ROLE.QUEEN
        self.__pg_mapping = OrderedDict()
        self.__NULL_GP = object()
        self.__current_pg = self.__NULL_GP
        self.SLEEPTIME = 1.0

    def valid_process_group(self, pg: ProcessGroup):
        return pg is not self.__NULL_GP and pg in self.__pg_mapping


# At most case, you are not allowed to modifed this list manually.
# FederatedWorld是底层的通讯抽象，World是对应的参数配置
__federated_world__: Dict[FederatedWorld, World] = OrderedDict()


class _Register(object):
    @classmethod
    def register_federated_world(cls, federated_world: FederatedWorld, world: World):
        if federated_world in __federated_world__:
            raise KeyError("Already registered.")
        else:
            __federated_world__[federated_world] = world

    @classmethod
    def deleted_federated_world(cls, federated_world: FederatedWorld):
        if federated_world in __federated_world__:
            if federated_world.is_initialized():
                print("Try to destroy all process group in federated world.")
                federated_world.destroy_process_group(
                    group=federated_world.WORLD)
            del __federated_world__[federated_world]

    @classmethod
    def is_registered(cls, federated_world: FederatedWorld) -> bool:
        return federated_world in __federated_world__

    def __iter__(self):
        return zip(__federated_world__.keys(), __federated_world__.values())

    @property
    def default_federated_world(cls) -> FederatedWorld:
        """ If not exists, return None
        """
        for fed_world in __federated_world__:
            return fed_world
        else:
            return None

    @property
    def default_world(cls) -> World:
        """If not exists, return None
        """
        for fed_world in __federated_world__:
            return __federated_world__[fed_world]
        else:
            return None

    def __len__(self):
        return len(__federated_world__)


register = _Register()
