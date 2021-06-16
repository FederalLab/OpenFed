# 在这里面定义所有数据类型与全局变量

import json
from enum import Enum, unique
from typing import Dict, List, TypeVar, Union

from torch import Tensor

_A = TypeVar("_A", bound='FedAddr')


class FedAddr(object):
    backend: str
    init_method: str = None
    world_size: int = 2
    rank: int = -1
    store = None
    group_name: str = ''

    def __init__(self,
                 backend: str,
                 init_method: str = None,
                 world_size: int = 2,
                 rank: int = -1,
                 store=None,
                 group_name: str = ''):
        """
        rank设置成-1的时候，会根据当前的身份自动推断。
        仅仅当你是在建立点对点的连接的时候，才会有效。也就是world size==2的时候。

        Initializes the default distributed process group, and this will also
        initialize the distributed package.

        There are 2 main ways to initialize a process group:
            1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
            2. Specify ``init_method`` (a URL string) which indicates where/how
            to discover peers. Optionally specify ``rank`` and ``world_size``,
            or encode all required parameters in the URL and omit them.

        If neither is specified, ``init_method`` is assumed to be "env://".


        Args:
            backend (str or Backend): The backend to use. Depending on
                build-time configurations, valid values include ``mpi``, ``gloo``,
                and ``nccl``. This field should be given as a lowercase string
                (e.g., ``"gloo"``), which can also be accessed via
                :class:`Backend` attributes (e.g., ``Backend.GLOO``). If using
                multiple processes per machine with ``nccl`` backend, each process
                must have exclusive access to every GPU it uses, as sharing GPUs
                between processes can result in deadlocks.
            init_method (str, optional): URL specifying how to initialize the
                                        process group. Default is "env://" if no
                                        ``init_method`` or ``store`` is specified.
                                        Mutually exclusive with ``store``.
            world_size (int, optional): Number of processes participating in
                                        the job. Required if ``store`` is specified.
            rank (int, optional): Rank of the current process (it should be a
                                number between 0 and ``world_size``-1).
                                Required if ``store`` is specified.
            store(Store, optional): Key/value store accessible to all workers, used
                                    to exchange connection/address information.
                                    Mutually exclusive with ``init_method``.
            timeout (timedelta, optional): Timeout for operations executed against
                the process group. Default value equals 30 minutes.
                This is applicable for the ``gloo`` backend. For ``nccl``, this is
                applicable only if the environment variable ``NCCL_BLOCKING_WAIT``
                or ``NCCL_ASYNC_ERROR_HANDLING`` is set to 1. When
                ``NCCL_BLOCKING_WAIT`` is set, this is the duration for which the
                process will block and wait for collectives to complete before
                throwing an exception. When ``NCCL_ASYNC_ERROR_HANDLING`` is set,
                this is the duration after which collectives will be aborted
                asynchronously and the process will crash. ``NCCL_BLOCKING_WAIT``
                will provide errors to the user which can be caught and handled,
                but due to its blocking nature, it has a performance overhead. On
                the other hand, ``NCCL_ASYNC_ERROR_HANDLING`` has very little
                performance overhead, but crashes the process on errors. This is
                done since CUDA execution is async and it is no longer safe to
                continue executing user code since failed async NCCL operations
                might result in subsequent CUDA operations running on corrupted
                data. Only one of these two environment variables should be set.
            group_name (str, optional, deprecated): Group name.

        To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
        on a system that supports MPI.
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.store = store
        self.group_name = group_name

    @classmethod
    def load_from_file(cls, file: str) -> List[_A]:
        """从文件加载的方式，创建的fedworld，不支持store的初始化方式！
        """
        with open(file, 'r') as f:
            fed_addr_dict_list = json.load(f)
        fed_addr_list = []
        for fed_addr_dict in fed_addr_dict_list:
            fed_addr_list.append(FedAddr(**fed_addr_dict))
        return fed_addr_list

    @classmethod
    def dump_to_file(cls, file: str, fed_addr_list: List[_A]):
        """
            fed_addr_list中的store不会被保存下来。因为不支持从这种方式初始化。
        """
        fed_addr_dict_list = []
        for fed_addr in fed_addr_list:
            fed_addr_dict_list.append(
                dict(backend=fed_addr.backend,
                     init_method=fed_addr.init_method,
                     world_size=fed_addr.world_size,
                     rank=fed_addr.rank,
                     group_name=fed_addr.group_name)
            )
        with open(file, "w") as f:
            json.dump(fed_addr_dict_list, f)

    def __expr__(self):
        # TODO: 更好的输出相关的信息
        return f"FedAddr: {self.init_method}"

    def as_dict(self):
        # 这个函数主要是为了让fedaddr支持解包操作，方便将其作为参数传入底层的方法。
        return dict(
            backend=self.backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.rank,
            store=self.store,
            group_name=self.group_name
        )


# 给定一个默认的fed addr地址，方便做实验验证的时候，不需要每次都指定地址。
default_fed_addr = FedAddr(
    backend="gloo", init_method='tcp://127.0.0.1:33298'
)


@unique
class APPROVED(Enum):
    GPU = "CPU"
    SYS = "GPU"
    ALL = "ALL"
    NONE = "None"


@unique
class ROLE(Enum):
    KING = True
    QUEEN = False


# 所有的操作都是由客户端向服务器端发送请求，服务器端只能应答请求。
# 当服务器完成应答后，会将客户端状态设置成ZOMBINE。
# 如果客户端下线，则程序状态改为OFFINE


@unique
class STATUS(Enum):
    PUSH = "True"  # 把数据推送到服务器
    PULL = "False"  # 从服务器拉取数据
    ZOMBINE = "None"  # 当客户端处于其他任何状态时，对于服务器来说，都是ZOMBINE的状态。
    OFFLINE = "OFFLINE"  # 当客户端不在线时，设置成OFFLINE。其余所有状态都表示客户端在线。
    # 因此，客户端程序退出时，应该记得调用相关函数，对状态进行设置。


PACKAGES = TypeVar(
    'PACKAGES', bound=Dict[str, Union[Tensor, Dict[str, Tensor]]])


# 如果DEBUG=True，那相关的程序会输出部分调试信息
# 会以更严格的方式，执行程序
DEBUG: bool = False


def debug():
    global DEBUG
    DEBUG = True


# 如果VERBOSE=True, 相关程序会输出一些日志
VERBOSE: bool = False


def verbose():
    global VERBOSE
    VERBOSE = True
