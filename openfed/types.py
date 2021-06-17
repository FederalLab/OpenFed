import json
import warnings
from enum import Enum, unique
from typing import Dict, List, TypeVar, Union

from prettytable import PrettyTable

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

    def __repr__(self):
        table = PrettyTable(
            ['backend', 'init_method', 'world_size', 'rank', 'store', 'group_name']
        )
        table.add_row(
            [self.backend, self.init_method, self.world_size,
                self.rank, self.store, self.group_name]
        )
        return "\n" + str(table)

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
    backend="gloo", init_method='tcp://localhost:1994', group_name="OpenFed"
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


# 如果DEBUG=True，那相关的程序会输出部分调试信息
# 会以更严格的方式，执行程序
DEBUG: bool = False


def debug():
    global DEBUG
    DEBUG = True


# 如果VERBOSE=True, 相关程序会输出一些日志
VERBOSE: bool = True


def verbose():
    global VERBOSE
    VERBOSE = True


def silence():
    global VERBOSE
    VERBOSE = False


def _check_state_keys(obj, keys: Union[str, List[str]], mode: str):
    keys = [keys] if isinstance(keys, str) else keys

    keys = keys if keys else getattr(obj, mode, None)

    if not keys and DEBUG:
        warnings.warn("Got empty keys")
    return keys


class Package(object):
    # 提供打包和解包statedict的能力
    state: Dict

    def pack_state(self, obj, keys: Union[str, List[str]] = None):
        """将obj中的state根据指定的key，pack到对应的数据流中。
        """
        keys = _check_state_keys(obj, keys, mode='package_key_list')
        if keys:
            for group in obj.param_groups:
                for p in group["params"]:
                    state = obj.state[p]
                    rdict = {k: state[k] for k in keys}
                    self.pack(p, rdict)

    def unpack_state(self, obj, keys: Union[str, List[str]] = None):
        keys = _check_state_keys(obj, keys, mode="unpackage_key_list")
        if keys:
            for group in obj.param_groups:
                for p in group["params"]:
                    state = obj.state[p]
                    rdict = {k: None for k in keys}
                    rdict = self.unpack(p, rdict)
                    state.update(rdict)
