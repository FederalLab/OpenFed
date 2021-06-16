import threading
from collections import OrderedDict
from typing import Any, Dict, overload

import openfed.utils.types as types
from torch.distributed.distributed_c10d import ProcessGroup


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
    # Dict[ProcessGroup, List[Package, Monitor, FederatedWorld]]
    _pg_mapping: Dict

    # 记录当前上层正在处理的pg是哪一个
    _NULL_GP: Any
    _current_pg: ProcessGroup

    # 我们并不希望这个参数被轻易修改，所以将它定义在这里，而不是CONSTANT里面。
    SLEEP_SHORT_TIME: float  # seconds
    SLEEP_LONG_TIME: float

    def __init__(self, ):
        self.ALIVE = True
        self.DEBUG = False
        self.VERBOSE = False

        # 不上传任何信息
        self.APPROVED = types.APPROVED.NONE
        self.ROLE = types.ROLE.QUEEN
        self._pg_mapping = OrderedDict()
        self._NULL_GP = object()
        self._current_pg = self._NULL_GP

        self.SLEEP_SHORT_TIME = .1
        self.SLEEP_LONG_TIME = 1.0

    def is_valid_process_group(self, pg: ProcessGroup):
        return pg is not self._NULL_GP and pg in self._pg_mapping

    # 添加一些锁，用于处理进程之间的同步问题
    # 提供一个context来做这件事
    joint_lock = threading.Lock()
