from collections import defaultdict
from typing import Any, Dict, List, Union

from bidict import bidict
from openfed.common import Package
from torch import Tensor

from ..core import FederatedWorld, ProcessGroup, World, gather_object
from .functional import Cypher


class Delivery(Package):

    pg: ProcessGroup
    federated_world: FederatedWorld
    world: World

    #! 注意！param对外始终保持id不变性，无论出现在哪里！只要是来自Package内部的数据，
    #! 那就始终都指向同一个内存区域！
    #! 只有当pull、push函数被调用以后以后，才会发生数据同步的操作！

    # key_tensor初始化之后id就不会发生改变
    key_tensor_bidict: bidict
    # package存储的是交互数据
    # 如果需要向外发送数据的话，会调用pack打包数据，然后传递给下层。
    # 如果需要从外接收数据的话，这个packages会被直接替换掉。
    # tensor_indexed_packages会重新使用key_tensor_dict，使其可以被原始的tensor正确索引。
    packages: Dict[str, Dict[str, Tensor]]
    # 默认情况下，packages是以str索引，这是为了方便处理从底层接收到的数据。
    # 我们提供了一个方法，使得其适合于用Tensor索引。
    # str标志是在所有的端中都是一致的。Tensor的id则并不是一致的。
    # 但是，使用tensor索引可以更灵活的和其他模块兼容。
    # 因此我们提供了两种方式.
    # tensor_indexed_packages

    # 字典中的每一个参数，都会传递给cypher。如果需要过滤某些参数，你可以直接修改cypher中实现的方法。
    # 在需要向外发送数据的情况下，cypher.pack函数会依次被调用
    # 在接收到外部传来的数据时，cypher.unpack函数会依次被调用
    _cypher_hooks: List[Cypher]

    def __init__(self) -> None:

        self.key_tensor_bidict = bidict()
        self.packages = defaultdict(dict)
        self._cypher_hooks = []

    def register_cypher(self, hook: Union[Cypher, List[Cypher]]):
        """添加一个hook或者一个hook list
        hook的pack和unpack函数会在发送、接收到数据时，自动调用。
        """
        if isinstance(hook, (list, tuple)):
            hook = (hook,)
        self._cypher_hooks.extend(hook)

    def key_tensor_map(self, key: str, tensor: Tensor):
        """将一个键值对加入到同步数据流中。
        key 不能是param，param是内部使用的键值。
        """
        if key in self.key_tensor_bidict or key == "param":
            raise KeyError(f"{key} already existed.")
        self.key_tensor_bidict[key] = tensor
        self.packages[key]["param"] = tensor

    def set_state_dict(self, state_dict: dict):
        """将state dict中的键值对加入到同步数据流中。
        """
        for k, v in state_dict.items():
            self.key_tensor_map(k, v)

    def key_name(self, t: Tensor) -> str:
        """返回一个tensor对应的字符串
        """
        return self.key_tensor_bidict.inverse[t]

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

        return rdict

    @property
    def tensor_indexed_packages(self):
        """返回一个由tensor索引的字典。当需要将接受的数据返回给aggretator处理的时候，调用此方法。
        """
        return {self.key_tensor(k): v for k, v in self.packages.items()}

    def reset(self):
        self.key_tensor_bidict = bidict()
        self.packages = defaultdict(dict)

    def pull(self) -> Dict[str, Dict[str, Tensor]]:
        """
        在调用这个函数之前，请告知另一端当前的状态
        从另一端拉取数据。
        如果此时是客户端，则会自动将拉取的数据中params参数以原址操作的方式，覆盖到key_tensor中。这是为了减少调用参数同步的额外操作。
        换言之，如果是客户端调用这个函数，则模型参数会被直接更新到模型中，而不是保留在数据流中。
        如果是服务器端，则不会有这个操作。
        """
        assert self.federated_world._get_group_size(
            self.pg) == 2, "Delivery is only designed for group with size 2"

        received = [None, None]
        rank = 0 if self.world.is_king() else 1
        other_rank = 1 if self.world.is_king() else 0

        gather_object(None, received, dst=rank, group=self.pg,
                      federated_world=self.federated_world)

        r_packages = received[other_rank]

        # unpack data here
        for hook in self._cypher_hooks:
            r_packages = {k: hook.unpack(k, v) for k, v in r_packages.items()}

        if self.world.is_queen():
            for k, v in r_packages.items():
                if 'param' in v:
                    self.key_tensor_bidict[k].data.copy_(v['param'])
        self.packages = r_packages

    def push(self) -> None:
        """向另一段发送数据。
        """
        assert self.federated_world._get_group_size(
            self.pg) == 2, "Delivery is only designed for group with size 2"

        rank = 1 if self.world.is_king() else 0

        # pack data here
        for hook in self._cypher_hooks:
            self.packages = {k: hook.pack(k, v)
                             for k, v in self.packages.items()}

        gather_object(self.packages, None, dst=rank,
                      group=self.pg, federated_world=self.federated_world)
