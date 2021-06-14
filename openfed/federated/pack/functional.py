from typing import Any
from torch import Tensor


class Function(object):  # type: ignore
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

    加入限制，保证pack的输出和unpack的输入是一致的。
    并且，不同的Function可以串联操作。
    """

    def pack(self, key: str, k: str, v: Tensor) -> Tensor:
        r"""
        Args:
            key: 在数据流中对应的键值
            k, v: 需要加入数据流中的键值对。
        返回处理之后的v。
        """
        raise NotImplementedError("You must implement the forward function for custom"
                                  " autograd.Function.")

    def unpack(self, key: str, k: str, v: Tensor) -> Tensor:
        r"""
        Args:
            key: 在数据流中对应的键值
            k, v: 需要加入数据流中的键值对。
        返回处理之后的v。
        """
        raise NotImplementedError("You must implement the backward function for custom"
                                  " autograd.Function.")
