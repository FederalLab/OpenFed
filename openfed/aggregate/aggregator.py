import functools
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Union

import torch
from torch import Tensor
from torch._six import container_abcs

from ..common import Package


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Aggregator."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Aggregator(Package):
    r"""Base class for all aggregators.

    Aggregator初始化阶段会接受一个参数列表。这个参数列表包含所有可能从客户端收集来的参数。
    收集来的参数会以{服务器参数：【接收参数列表】}的方式以字典的方式存储。
    接收的参数中一个服务器参数可能会绑定多个相关的参数。
        记住！！接收参数是客户端更新后的参数，不是客户端累计的更新量！
    每一次聚合阶段，我们会遍历所有的初始参数列表。如果能够找到对应参数，则进行聚合操作，否则跳过。
    在聚合阶段，所有的子类都应当遵循以下规则：
        1. 如果参数属性requires_grad=True，则将更新的参数记录在相应的grad属性中，而不直接修改服务器参数。
        2. 如果参数属性requires_grad=False，则直接覆盖服务器参数。

    每次传入的数据，除了参数字典外，还会顺带传入一个信息字典。信息字典中包含了一些来自客户端的基本数据。
    信息字典中的每一个条目，如果符合以下格式，则会被自动进行聚合：
        training_samples:
            value: 1000
            reduce_op: sum, 
    其余条目则会被忽略。

    defaults参数中定义了聚合过程中所必须包含的信息和其初始值，例如train_samples=0.
    其中，初始值在大部分条件下不会被使用到。如果返回的信息字典中缺乏defaults中定义的关键词，则会发生错误。

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of aggregation
            options (used when a parameter group doesn't specify them).
    """

    # 列表中的参数会被打包到package中
    package_key_list: List[str] = None

    # 列表中的参数会从package中解压出来
    unpackage_key_list: List[str] = None

    _hook_for_auto_reduce_infos: List[Callable]
    _received_infos: List[Dict]

    def __init__(self,
                 params,
                 defaults: Dict,
                 info_keys: List[str],
                 aux_keys: List[str],
                 lagecy: bool = False):
        """
            info_keys: 表示在聚合的过程中所需要涉及到的其他数据。
            aux_keys: 进行聚合操作时，需要的额外的数据，这些数据会在调用zero_grad之后被清除。
            lagecy: True表示在接收到新的数据后，直接缓存下来。False表示尽可能与之前的数据合并，以减少内存占用。
        """
        # reset keys in state will be deleted once reset() called.
        self.lagecy = lagecy

        # add info_keys to defaults
        defaults['info_keys'] = info_keys
        defaults['aux_keys'] = aux_keys
        defaults['lagecy'] = lagecy

        self.defaults = defaults

        self._hook_for_profile()

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the aggregator should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("aggregator got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        self._hook_for_auto_reduce_infos = []
        self._received_infos = []

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._hook_for_profile()  # To support multiprocessing pickle/unpickle.

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def _hook_for_profile(self):
        self._zero_grad_profile_name = "Aggregator.zero_grad#{}.zero_grad".format(
            self.__class__.__name__)

        def profile_hook_step(func):

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                obj, *_ = args
                profile_name = "Aggregator.step#{}.step".format(
                    obj.__class__.__name__)
                with torch.autograd.profiler.record_function(profile_name):
                    return func(*args, **kwargs)
            return wrapper

        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = profile_hook_step(self.__class__.step)
            self.__class__.step.hooked = True

    def state_dict(self):
        r"""Returns the state of the aggregator as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current aggregation state. Its content
            differs between aggregator classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the aggregator state.

        Args:
            state_dict (dict): aggregator state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of aggregator's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        除了清空梯度以外，这个函数还会清空aggregator的内部缓存。
        如果是在服务器端使用，请务必调用aggregator.zero_grad()来清除梯度信息，而不是optimzier.zero_grad()。

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` aggregators have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        if not hasattr(self, "_zero_grad_profile_name"):
            self._hook_for_profile()

        # Clear buffers
        self._received_infos = []

        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            p.grad.zero_()
                    # Clear buffer
                    state = self.state[p]
                    for k in group["aux_keys"]:
                        if k in state:
                            del state[k]

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Aggregator` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Aggregator` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific aggregation options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('aggregator parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("aggregator can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required aggregation parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("aggregator contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def _check_defaults_keys(self, info_keys: List[str], received_info: Dict):
        """检查返回的信息中是否有缺失了必要信息。
        """
        for key in info_keys:
            if key not in received_info:
                raise KeyError(f"{key} is needed, but not given.")

    def register_hook_for_auto_reduce_infos(self, func: Callable):
        """符合特定格式的键值对会被自动聚合。

        func会接收聚合的List[Dict]数据，计算出想要的结果，然后返回。
        """
        self._hook_for_auto_reduce_infos.append(func)

    def _apply_hook_for_auto_reduce(self) -> List[Any]:
        returns = []
        for fn in self._hook_for_auto_reduce_infos:
            returns.append(fn(self._received_infos))
        return returns

    def aggregate(self) -> Dict:
        r"""Performs a single aggregation step (parameter update).

        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most aggregators.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        该函数调用后，所有参数的grad属性被正确设置。
        该函数会根据使用的是merge还是append的方式，来计算出最终正确的结果。
        step函数会调用hook_for_auto_recude_info，返回处理结果。
        """
        for group in self.param_groups:
            lagecy = group['lagecy']
            for p in group["params"]:
                if lagecy:
                    self._stack_aggregate(p, group=group)
                else:
                    self._merge_aggregate(p, group=group)

        return self._apply_hook_for_auto_reduce()

    def step(self, received_params: Dict[str, Union[Tensor, Dict[str, Tensor]]], received_info: Dict):
        """追加一个新的数据，并进行相关检查。
        """
        for group in self.param_groups:
            self._check_defaults_keys(group['info_keys'], received_info)
            lagecy = group['lagecy']
            for p in group["params"]:
                if p in received_params:
                    if lagecy:
                        self.stack(p, received_params[p],
                                   received_info=received_info, group=group)
                    else:
                        self.merge(p, received_params[p],
                                   received_info=received_info, group=group)
        self._received_infos.append(received_info)

    def merge(self, p: Tensor, r_p: Dict[str, Tensor], received_info: Dict, group: Dict) -> Any:
        """采用融合的方式，吸收新的客户端参数。始终只需要保存一个参数副本。
        这种方式会更加节省内存空间，但是会损失每一个client的具体信息。因此部分算法可能不支持此设置。
        如果你的算法支持这种方式，请实现它。
        返回当前aggregator的一些状态字典。
        """
        raise NotImplementedError

    def _merge_aggregate(self, p: torch.Tensor, group: Dict) -> None:
        raise NotImplementedError

    def stack(self, p: Tensor, r_p: Dict[str, Tensor], received_info: Dict, group: Dict) -> Any:
        """采用追加的方式，吸收新的客户端参数。每次都直接把参数保存到内存中。
        这种方式会占据大量的内存空间，但是可以保存所有的参数细节。
        所有的算法，都应该支持这种方式。
        返回当前aggregator的一些状态字典。
        """
        raise NotImplementedError

    def _stack_aggregate(self, p: torch.Tensor, group: Dict) -> None:
        raise NotImplementedError
