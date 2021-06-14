import torch
from torch.optim import Optimizer


class ElasticAux(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    
    这个特殊的优化器是配合ElasticAggregater使用的，它要求在客户端统计一个参数的重要性参数。
    这个特殊的优化器不会更新参数，只是负责计算参数的重要性，并且保存在state中。
    使用如下：
    Example:
        >>> elastic_aux = ElasticAux(net.parameters(), momentum=0.9)
        >>> while:
        >>>     elastic_aux.zero_grad()
        >>>     # 使用网络的输出和0之间的MSE LOSS来计算梯度。
        >>>     MSE(net(input), zeros).backward()
        >>>     elastic_aux.step()
        >>> # 如果你想重复使用，请调用clear函数清除之前的计算的重要性参数。
        >>> elastic_aux.clear()

    """

    def __init__(self, params, momentum=0.9):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.abs()

                state = self.state[p]

                if 'importance' not in state:
                    state["importance"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                state["importance"].mul_(momentum).add_(grad, alpha=1-momentum)

        return loss

    def clear(self):
        """清除掉之前计算出来的结果。
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'importance' in state:
                    del state['importance']
