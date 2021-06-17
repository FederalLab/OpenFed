from abc import abstractmethod
from typing import Any, Dict

from torch import Tensor


class Cypher(object):
    r"""Records operation history and defines formulas for differentiating ops.

    See the Note on extending the autograd engine for more details on how to use
    this class: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd

    Every operation performed on :class:`Tensor` s creates a new function
    object, that performs the computation, and records that it happened.
    The history is retained in the form of a DAG of functions, with edges
    denoting data dependencies (``input <- output``). Then, when backward is
    called, the graph is processed in the topological ordering, by calling
    :func:`backward` methods of each :class:`Function` object, and passing
    returned gradients on to next :class:`Function` s.

    Normally, the only way users interact with functions is by creating
    subclasses and defining new operations. This is a recommended way of
    extending torch.autograd.

    Examples::

        >>> class Exp(Function):
        >>>
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result
        >>>
        >>> #Use it by calling the apply method:
        >>> output = Exp.apply(input)

        加密和解密过程永远在不同的端进行。
        如果解密过程中需要涉及加密过程的一些数据，请不要直接设置成属性！而是把它放到Dict中一并返回。

        如果多个保密模块被叠加使用，那么加密过程和解密过程将逆序进行。也就是：先加密，后解密。
    """

    @abstractmethod
    def encrypt(self, key: str, value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        r"""
        Args:
            key 用于标记value，可以根据key来做一些选择性处理。
            value是一个字典，这个字典的值可能会被改变之后返回。
        """
        raise NotImplementedError("You must implement the forward function for custom"
                                  " autograd.Function.")

    @abstractmethod
    def decrypt(self, key: str, value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        r"""
            这里接收来自对方pack过的数据，现在进行unpack。
        """
        raise NotImplementedError("You must implement the backward function for custom"
                                  " autograd.Function.")
