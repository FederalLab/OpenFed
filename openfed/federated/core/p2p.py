import contextlib

from openfed.federated import register

from .backend import Backend
from .federated_c10d import ProcessGroupNCCL
from .functional import _check_op, _check_single_tensor


@contextlib.contextmanager
def _batch_p2p_manager(backend):
    if backend == Backend.NCCL:
        ProcessGroupNCCL._group_start()
    try:
        yield
    finally:
        if backend == Backend.NCCL:
            ProcessGroupNCCL._group_end()


class P2POp(object):
    """
    A class to build point-to-point operations for ``batch_isend_irecv``.

    This class builds the type of P2P operation, communication buffer, peer rank,
    Process Group group, and tag. Instances of this class will be passed to
    ``batch_isend_irecv`` for point-to-point communications.

    Args:
        op (callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``torch.distributed.isend`` or
            ``torch.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int): Destination or source rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with recv.
    """

    def __init__(self, op, tensor, peer, group=None, tag=0):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag

    def __new__(cls, op, tensor, peer, group=None, tag=0):
        _check_op(op)
        _check_single_tensor(tensor, "tensor")
        return object.__new__(cls)


def batch_isend_irecv(p2p_op_list,
                      federated_world=None):
    """
    Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the operations in p2p_op_list and return the corresponding
    requests. NCCL and Gloo backend are currently supported.

    Args:
        p2p_op_list: A list of point-to-point operations(type of each operator is
            ``torch.distributed.P2POp``). The order of the isend/irecv in the list
            matters and it needs to match with corresponding isend/irecv on the
            remote end.

    Returns:
        A list of distributed request objects returned by calling the corresponding
        op in the op_list.

    Examples:
        >>> send_tensor = torch.arange(2) + 2 * rank
        >>> recv_tensor = torch.randn(2)
        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)
        >>> recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank + 1)%world_size)
        >>> reqs = batch_isend_irecv([send_op, recv_op])
        >>> for req in reqs:
        >>>     req.wait()
        >>> recv_tensor
        tensor([2, 3])     # Rank 0
        tensor([0, 1])     # Rank 1

    .. note:: Note that when this API is used with the NCCL PG backend, users must set
        the current GPU device with `torch.cuda.set_device`, otherwise it will
        lead to unexpected hang issues.
    """
    if federated_world is None:
        federated_world = register.default_federated_world
    federated_world._check_p2p_op_list(p2p_op_list)
    backend = federated_world.get_backend(p2p_op_list[0].group)
    reqs = []
    with _batch_p2p_manager(backend):
        for p2p_op in p2p_op_list:
            op = p2p_op.op
            tensor = p2p_op.tensor
            peer = p2p_op.peer
            curr_group = p2p_op.group
            tag = p2p_op.tag

            ret = op(tensor, peer, curr_group, tag)

            if ret is not None:
                reqs.append(ret)
    return reqs
