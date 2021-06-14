from typing import Dict, List, Union
from .functional import Function
import openfed
from bidict import bidict
from openfed import PACKAGES
from torch import Tensor
from torch.optim import Optimizer


class Package(object):

    #! 注意！param对外始终保持id不变性，无论出现在哪里！只要是来自Package内部的数据，
    #! 那就始终都指向同一个内存区域！

    # key_tensor初始化之后id就不会发生改变
    key_tensor_bidict: bidict
    # package存储的是交互数据
    # 如果需要向外发送数据的话，会调用pack打包数据，然后传递给下层。
    # 如果需要从外接收数据的话，这个packages会被直接替换掉。
    # tensor_indexed_packages会重新使用key_tensor_dict，使其可以被原始的tensor正确索引。
    packages: PACKAGES

    # 字典中的每一个参数，都会传递给function。如果需要过滤某些参数，你可以直接修改function中实现的方法。
    # 在需要向外发送数据的情况下，function.pack函数会依次被调用
    # 在接收到外部传来的数据时，function.unpack函数会依次被调用
    _function_hooks: List[Function]

    def __init__(self, state_dict: Dict[str, Tensor]):
        """receive state_dict and build <key, tensor> map.

        .. note::
            Once state_dict specified, the tensor id is not allowed to changed. 
            If have to, please use a new Package instead.

        一定要注意tensor是在全局保持一致的！
        """
        self.key_tensor_bidict = bidict(state_dict)
        self.reset()

    def reset(self):
        """reset stacked packages. 
        """
        # 不要使用clear函数，packages是直接交付的数据，如果是用clear函数的话，会影响原址操作！
        self.packages = dict(self.key_tensor_bidict)

    def state_dict(self):
        # 这个方法一般不会被使用到
        # 因为state_dict始终是保持id不变的。
        return dict(self.key_tensor_bidict)

    def key_name(self, t: Tensor):
        return self.key_tensor_bidict.inverse[t]

    def key_tensor(self, key: str):
        return self.key_tensor_bidict[key]

    def append(self, p: Tensor, rdict: Dict[str, Tensor]):
        key_name = self.key_name(p)
        package = self.packages.get(key_name)

        if isinstance(package, dict):
            package.update(rdict)
        else:
            assert isinstance(package, Tensor)
            # 将原来的state_dict的对应键值转换成字典。
            package = {"param": package}
            package.update(rdict)

        self.packages[key_name] = package

    @property
    def tensor_index_packages(self):
        """返回一个可以由tensor索引的字典
        """
        return {self.key_tensor(key_name): v for key_name, v in self.packages.items()}

    ####
    # pack/unpack函数主要是用来打包数据的，提供给上层使用。
    ####

    def pack_optimizer_state(self, optimizer: Optimizer, state_keys: Union[List[str], str]):
        if isinstance(state_keys, str):
            state_keys = [state_keys]

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = optimizer.state[p]
                    rdict = {key: state[key] for key in state_keys}
                    self.append(p, rdict)

    def unpack_optimizer_state(self, optimizer: Optimizer, state_keys: Union[str, List[str]]):
        if isinstance(state_keys, str):
            state_keys = [state_keys]

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = optimizer.state[p]

                    key_name = self.key_name(p)
                    rdict = self.packages[key_name]
                    assert isinstance(
                        rdict, dict), "Must be a dict contains parameter."
                    # load the value.
                    for key in state_keys:
                        state[key] = rdict[key]

    def pack_aggregater_state(self, aggregator, state_keys: Union[str, List[str]]):
        if isinstance(state_keys, str):
            state_keys = [state_keys]

        for group in aggregator.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = aggregator.state[p]
                    rdict = {key: state[key] for key in state_keys}
                    self.append(p, rdict)

    def unpack_aggregater_state(self, aggregator, state_keys: Union[str, List[str]]):
        if isinstance(state_keys, str):
            state_keys = [state_keys]

        for group in aggregator.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = aggregator.state[p]

                    key_name = self.key_name(p)
                    rdict = self.packages[key_name]
                    assert isinstance(
                        rdict, dict), "Must be a dict contains parameter."
                    # load the value.
                    for key in state_keys:
                        state[key] = rdict[key]

    ####
    # handout/handin函数主要是用来和底层的通信模块交互的。
    ####
    def handout(self) -> PACKAGES:
        """将本模块打包好的数据，以符合要求的字典格式交付给通信层。
        """
        return self.packages

    def handin(self, packages: PACKAGES):
        """接收到的数据包，可能数据类型不对，这里需要对齐。另外，接收到的数据包，某些键值可能会丢失，注意容错。
        多余的未注册的键值，则直接忽略。
        如果是客户端，会直接将接收到的param数据拷贝到数据中。
        """
        # 自动清理一下缓存的package
        self.reset()

        for key in self.key_tensor_bidict.keys():
            if key in packages:
                if isinstance(packages[key], dict):
                    package = packages[key]
                    p = self.key_tensor(key)

                    # 确保所有的键值在正确的设备上
                    new_package = {k: v.type_as(p) for k, v in package.items()}

                    if openfed.is_queen():
                        # 如果是客户端，直接把param属性拷贝到tensor上！
                        p.data.copy_(new_package['param'])
                        # 然后用p更新字典
                        new_package['param'] = p
                    else:
                        # 服务器端不允许直接覆盖原始参数！！！
                        pass
                else:
                    param = packages[key]
                    assert isinstance(param, Tensor)

                    p = self.key_tensor(key)
                    param = param.type_as(p)

                    if openfed.is_queen():
                        p.data.copy_(param)
                    new_package = p
                self.packages[key] = new_package
