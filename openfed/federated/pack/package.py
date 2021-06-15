from collections import defaultdict
from typing import Any, Dict, List, Union
import warnings

from bidict import bidict
from openfed.aggregate.aggregator import Aggregator
from openfed.federated.core.federated_c10d import FederatedWorld, ProcessGroup
from openfed.federated.register import World
from openfed.utils.types import PACKAGES
from torch import Tensor
from torch.optim import Optimizer

from .deliver import Delivery
from .functional import Function


class Package(object):

    #! 注意！param对外始终保持id不变性，无论出现在哪里！只要是来自Package内部的数据，
    #! 那就始终都指向同一个内存区域！
    #! 只有当pull、push函数被调用以后以后，才会发生数据同步的操作！

    # key_tensor初始化之后id就不会发生改变
    key_tensor_bidict: bidict
    # package存储的是交互数据
    # 如果需要向外发送数据的话，会调用pack打包数据，然后传递给下层。
    # 如果需要从外接收数据的话，这个packages会被直接替换掉。
    # tensor_indexed_packages会重新使用key_tensor_dict，使其可以被原始的tensor正确索引。
    packages: PACKAGES
    # 默认情况下，packages是以str索引，这是为了方便处理从底层接收到的数据。
    # 我们提供了一个方法，使得其适合于用Tensor索引。
    # str标志是在所有的端中都是一致的。Tensor的id则并不是一致的。
    # 但是，使用tensor索引可以更灵活的和其他模块兼容。
    # 因此我们提供了两种方式.
    # tensor_indexed_packages

    # 字典中的每一个参数，都会传递给function。如果需要过滤某些参数，你可以直接修改function中实现的方法。
    # 在需要向外发送数据的情况下，function.pack函数会依次被调用
    # 在接收到外部传来的数据时，function.unpack函数会依次被调用
    __function_hooks: List[Function]

    # 负责数据传送
    deliver: Delivery

    def __init__(self, pg: ProcessGroup, federated_world: FederatedWorld, world: World) -> None:
        self.deliver = Delivery(pg, federated_world, world)
        self.key_tensor_bidict = bidict()
        self.packages = defaultdict(dict)

    def register_hook(self, hook: Union[Function, List[Function]]):
        """添加一个hook或者一个hook list
        hook的pack和unpack函数会在发送、接收到数据时，自动调用。
        """
        if isinstance(hook, (list, tuple)):
            hook = (hook,)
        self.__function_hooks.extend(hook)

    def key_tensor_map(self, key: str, tensor: Tensor):
        """将一个键值对加入到同步数据流中。
        key 不能是param，param是内部使用的键值。
        """
        if key in self.key_tensor_bidict or key == "param":
            raise KeyError(f"{key} already existed.")
        self.key_tensor_bidict[key] = tensor

    def state_dict_map(self, state_dict: dict):
        """将state dict中的键值对加入到同步数据流中。
        """
        for k, v in state_dict.items():
            self.key_tensor_map(k, v)

    def key_name(self, t: Tensor) -> str:
        """返回一个tensor对应的字符串
        """
        return self.key_tensor_bidict.inverse(t)

    def key_tensor(self, key: str) -> Tensor:
        """返回一个字符串对应的tensor。
        """
        return self.key_tensor_bidict.get(key)

    def pack(self, key: Union[str, Tensor], rdict: Dict[str, Tensor]):
        """给定数据流中的一个key（可以是tensor也可以是str），将rdict中的内容，更新到数据流中。将覆盖数据。
        """
        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        # pack data
        for hook in self.__function_hooks:
            rdict = {k: hook.pack(key, k, v) for k, v in rdict.items()}

        package = self.packages.get(key)

        package.update(rdict)

    def unpack(self, key: Union[str, Tensor], rdict: Dict[str, Any]):
        """给定数据流中的一个key（可以是Tensor也可以是str），将数据流中rdict对应的键值加载到rdict中。
        """
        if not isinstance(key, str):
            assert isinstance(key, Tensor)
            key = self.key_name(key)

        package = self.packages.get(key)
        rdict = {k: package[k] for k in rdict}

        # unpack data
        for hook in self.__function_hooks:
            rdict = {k: hook.unpack(key, k, v) for k, v in rdict.items()}

        return rdict

    def pack_state(self, obj: Union[Optimizer, Aggregator], keys: Union[str, List[str]] = None):
        """将obj中的state根据指定的key，pack到对应的数据流中。
        """
        if isinstance(keys, str):
            keys = [keys]

        if keys is None:
            if hasattr(obj, "package_key_list"):
                keys = obj.package_key_list
            else:
                raise ValueError("Got empty keys")

        if len(keys) == 0:
            # Empty keys
            warnings.warn("Got empty keys")
            return

        for group in obj.param_groups:
            for p in group["params"]:
                state = obj.state[p]
                rdict = {k: state[k] for k in keys}
                self.pack(p, rdict)

    def unpack_state(self, obj: Union[Optimizer, Aggregator], keys: Union[str, List[str]] = None):
        if isinstance(keys, str):
            keys = [keys]

        if keys is None:
            if hasattr(obj, "unpackage_key_list"):
                keys = obj.unpackage_key_list
            else:
                raise ValueError("Got empty keys")

        if len(keys) == 0:
            # Empty keys
            warnings.warn("Got empty keys")
            return

        for group in obj.param_groups:
            for p in group["params"]:
                state = obj.state[p]
                rdict = {k: None for k in keys}
                rdict = self.unpack(p, rdict)
                state.update(rdict)

    @property
    def tensor_index_packages(self):
        """返回一个由tensor索引的字典。当需要将接受的数据返回给aggretator处理的时候，调用此方法。
        """
        return {self.key_tensor(k): v for k, v in self.packages.items()}

    def pull(self):
        """
        在调用这个函数之前，请通过monitor，告知另一端当前的状态
        从另一端拉取数据。
        如果此时是客户端，则会自动将拉取的数据中params参数以原址操作的方式，覆盖到key_tensor中。这是为了减少调用参数同步的额外操作。
        换言之，如果是客户端调用这个函数，则模型参数会被直接更新到模型中，而不是保留在数据流中。
        如果是服务器端，则不会有这个操作。
        """
        # 拉取数据
        r_packages = self.deliver.pull()

        if self.deliver.world.is_queen():
            for k, v in r_packages.items():
                if 'param' in v:
                    self.key_tensor_bidict[k].data.copy_(v['param'])
        self.packages = r_packages

    def push(self):
        """
        在调用这个函数之前，请通过monitor，告知另一端当前的状态。
        将数据推送到另一端。
        该函数会将key中的tensor以param的键值，存储在数据流中。
        """
        for k, v in self.packages.items():
            v['param'] = self.key_tensor_bidict[k]

        self.deliver.push(self.packages)

    def reset(self):
        """重置这个类中的状态，但是保留deliver。
        一般情况下不需要调用这个函数。
        """
        self.key_tensor_bidict = bidict()
        self.packages = defaultdict(dict)
