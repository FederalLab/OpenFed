from ..utils.types import *
from dataclasses import dataclass
from .core.federated_c10d import FederatedWorld
from collections import OrderedDict


@dataclass
class World(object):
    # 标志openfed系统是否正在运行。
    # 当ALIVE为False时，程序中相关的一些后台进程会自动退出。
    # 保证程序平稳运行
    ALIVE = True

    # TODO: 设计一个统一的方式，退出后台线程。
    # 后台线程包括Thread、Generator

    def killed():
        global ALIVE
        ALIVE = False

    # 如果设置为True的话，那相关的程序会输出相应的调试信息。
    DEBUG = False

    # TODO：设计一个统一的方式，对调试信息格式化输出。

    def debug():
        global DEBUG
        DEBUG = True

    # 如果设置为True的话，相关的程序会输出运行状态
    VERBOSE = False

    # TODO：设置一个统一的方式，进行可视化输出

    def verbose():
        global VERBOSE
        VERBOSE = True

    # 是否允许openfed传输系统相关信息以方便对方监视自己当前运行状态。
    _APPROVED = APPROVED.ALL

    # 大部分情况下是QUEEN，客户端。毕竟服务器端就只会有一个。
    _ROLE = ROLE.QUEEN

    def set_king():
        global _ROLE
        _ROLE = ROLE.KING

    def set_queen():
        global _ROLE
        _ROLE = ROLE.QUEEN

    def who_am_i():
        print(f"I am the {_ROLE}")

    def is_king():
        return _ROLE == ROLE.KING

    def is_queen():
        return _ROLE == ROLE.QUEEN

    @overload
    def set_openfed_state(state: APPROVED):
        """这是用来设置权限的。
        """

    @overload
    def set_openfed_state(state: ROLE):
        """这是用来设置身份的。
        """

    def set_openfed_state(state):
        """这里才是函数的实现。
        """
        # 根据不同的state类型，设置相对应的状态。
        if isinstance(state, ROLE):
            global _ROLE
            _ROLE = state
        elif isinstance(state, APPROVED):
            global _APPROVED
            _APPROVED = state
        else:
            raise NotImplementedError


# At most of case, you are not allowed to modifed this list manually.
__federated_world__: Dict[FederatedWorld, Any] = OrderedDict()


class _Register(object):
    @classmethod
    def register_federated_world(cls, federated_world: FederatedWorld, description: Any = None):
        if federated_world in __federated_world__:
            raise KeyError("Already registered.")
        else:
            __federated_world__[federated_world] = description

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
        return self

    def __next__(self) -> List[Union[FederatedWorld, Any]]:
        while len(__federated_world__) > 0:
            for fed_world in __federated_world__.items():
                return fed_world
        else:
            raise StopIteration

    @property
    def default_federated_world(cls) -> FederatedWorld:
        """ If not exists, return None
        """
        if len(__federated_world__) > 0:
            for fed_world in __federated_world__:
                return fed_world
        else:
            return None


register = _Register()
