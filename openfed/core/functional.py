# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch._C._distributed_c10d import (AllreduceCoalescedOptions,
                                        AllreduceOptions, AllToAllOptions,
                                        BroadcastOptions, GatherOptions,
                                        ReduceOp, ReduceOptions,
                                        ReduceScatterOptions, ScatterOptions)
from torch.distributed.distributed_c10d import (
    Backend, _batch_p2p_manager, _check_single_tensor, _check_tensor_list,
    _object_to_tensor, _tensor_to_object, supports_complex)


def isend(tensor, dst, group=None, tag=0, country=None):
    """
    Sends a tensor asynchronously.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        country (Country): The process group belongs to.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    assert country

    _check_single_tensor(tensor, "tensor")
    if country._rank_not_in_group(group):
        return

    if group is None or group is country.WORLD:
        default_pg = country._get_default_group()
        return default_pg.send([tensor], dst, tag)
    else:
        dst = country._get_group_rank(group, dst)
        return group.send([tensor], dst, tag)


def irecv(
    tensor,
    src,
    group=None,
    tag=0,
    country=None,
):
    """
    Receives a tensor asynchronously.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int): Source rank. Will receive from any
            process if unspecified(None). 
        country (Country): The process group belongs to.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    assert country
    _check_single_tensor(tensor, "tensor")
    if country._rank_not_in_group(group):
        return

    if group is None or group is country.WORLD:
        pg = country._get_default_group()
    else:
        pg = group

    if src is None:
        return pg.recv_anysource([tensor], tag)
    else:
        if pg is country.WORLD:
            return pg.recv([tensor], src, tag)
        else:
            src = country._get_group_rank(pg, src)

            return pg.recv([tensor], src, tag)


def send(
    tensor,
    dst,
    group=None,
    tag=0,
    country=None,
):
    """
    Sends a tensor synchronously.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        country (Country): The process group belongs to.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv

    """
    assert country
    _check_single_tensor(tensor, "tensor")
    if country._rank_not_in_group(group):
        return

    if group is None or group is country.WORLD:
        default_pg = country._get_default_group()
        default_pg.send([tensor], dst, tag).wait()
    else:
        dst = country._get_group_rank(group, dst)
        group.send([tensor], dst, tag).wait()


def recv(
    tensor,
    src,
    group=None,
    tag=0,
    country=None,
):
    """
    Receives a tensor synchronously.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank. Will receive from any
            process if unspecified(None).
        country (Country): The process group belongs to.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send

    Returns:
        Sender rank
        -1, if not part of the group

    """
    assert country
    _check_single_tensor(tensor, "tensor")
    if country._rank_not_in_group(group):
        return -1

    if group is None:
        pg = country._get_default_group()
    else:
        pg = group

    if src is None:
        work = pg.recv_anysource([tensor], tag)
        work.wait()
        src_rank = work._source_rank()
        if group is None or group is country.WORLD:
            return src_rank
        else:
            return country._get_global_rank(pg, src_rank)
    else:
        if group is None or group is country.WORLD:
            pg.recv([tensor], src, tag).wait()
        else:
            src = country._get_group_rank(pg, src)
            pg.recv([tensor], src, tag).wait()
        return src


def broadcast_multigpu(
    tensor_list,
    src,
    group=None,
    async_op=False,
    src_tensor=0,
    country=None,
):
    """
    Broadcasts the tensor to the whole group with multiple GPU tensors
    per node.

    ``tensor`` must have the same number of elements in all the GPUs from
    all processes participating in the collective. each tensor in the list must
    be on a different GPU

    Only nccl and gloo backend are currently supported
    tensors should only be GPU tensors

    Args:
        tensor_list (List[Tensor]): Tensors that participate in the collective
            operation. If ``src`` is the rank, then the specified ``src_tensor``
            element of ``tensor_list`` (``tensor_list[src_tensor]``) will be
            broadcast to all other tensors (on different GPUs) in the src process
            and all tensors in ``tensor_list`` of other non-src processes.
            You also need to make sure that ``len(tensor_list)`` is the same
            for all the distributed processes calling this function.

        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        src_tensor (int, optional): Source tensor rank within ``tensor_list``

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    assert country
    if country._rank_not_in_group(group):
        return

    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = src_tensor

    if group is None or group is country.WORLD:
        default_pg = country._get_default_group()
        work = default_pg.broadcast(tensor_list, opts)
    else:
        group_src_rank = country._get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.broadcast(tensor_list, opts)
    if async_op:
        return work
    else:
        work.wait()


def broadcast(tensor, src, group=None, async_op=False, country=None):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    assert country
    _check_single_tensor(tensor, "tensor")
    if country._rank_not_in_group(group):
        return

    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = 0

    if group is None or group is country.WORLD:
        default_pg = country._get_default_group()
        work = default_pg.broadcast([tensor], opts)
    else:
        group_src_rank = country._get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.broadcast([tensor], opts)
    if async_op:
        return work
    else:
        work.wait()


def all_reduce_multigpu(tensor_list,
                        op=ReduceOp.SUM,
                        group=None,
                        async_op=False,
                        country=None):
    r"""
    Reduces the tensor data across all machines in such a way that all get
    the final result. This function reduces a number of tensors on every node,
    while each tensor resides on different GPUs.
    Therefore, the input tensor in the tensor list needs to be GPU tensors.
    Also, each tensor in the tensor list needs to reside on a different GPU.

    After the call, all ``tensor`` in ``tensor_list`` is going to be bitwise
    identical in all processes.

    Complex tensors are supported.

    Only nccl and gloo backend is currently supported
    tensors should only be GPU tensors

    Args:
        tensor list (List[Tensor]): List of input and output tensors of
            the collective. The function operates in-place and requires that
            each tensor to be a GPU tensor on different GPUs.
            You also need to make sure that ``len(tensor_list)`` is the same for
            all the distributed processes calling this function.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    assert country
    if country._rank_not_in_group(group):
        return

    tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in tensor_list
    ]

    opts = AllreduceOptions()
    opts.reduceOp = op
    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.allreduce(tensor_list, opts)
    else:
        work = group.allreduce(tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def all_reduce(tensor,
               op=ReduceOp.SUM,
               group=None,
               async_op=False,
               country=None):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4, 6]) # Rank 0
        tensor([4, 6]) # Rank 1

        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j]) # Rank 0
        tensor([3.+3.j, 4.+4.j]) # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4.+4.j, 6.+6.j]) # Rank 0
        tensor([4.+4.j, 6.+6.j]) # Rank 1

    """
    assert country
    _check_single_tensor(tensor, "tensor")
    if country._rank_not_in_group(group):
        return

    if tensor.is_complex():
        if not supports_complex(op):
            raise RuntimeError(
                f"all_reduce does not support {op} on complex tensors")
        tensor = torch.view_as_real(tensor)

    opts = AllreduceOptions()
    opts.reduceOp = op
    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.allreduce([tensor], opts)
    else:
        work = group.allreduce([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


def all_reduce_coalesced(tensors,
                         op=ReduceOp.SUM,
                         group=None,
                         async_op=False,
                         country=None):
    """
    WARNING: at this time individual shape checking is not implemented across nodes.
    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the allreduce
    operation will proceed without complaint and return erroneous outputs. This lack
    of shape checking results in significant performance improvements but users of this
    function should take extra care to ensure that each node passes in tensors whose
    shapes match across nodes.

    Reduces each tensor in tensors (residing on the same device) across all machines
    in such a way that all get the final result.

    After the call each tensor in tensors is going to bitwise identical
    in all processes.

    Complex tensors are supported.

    Args:
        tensors (List[Tensor]): Input and output of the collective. The function
            operates in-place.
        op (Optional[ReduceOp]): One of the values from
            ``torch.distributed.ReduceOp`` enum. Specifies an operation used for
            element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (Optional[bool]): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert country
    _check_tensor_list(tensors, "tensor")
    if country._rank_not_in_group(group):
        return

    if any([t.is_complex() for t in tensors]) and not supports_complex(op):
        raise RuntimeError(
            f"all_reduce does not support {op} on complex tensors")

    tensors = [
        t if not t.is_complex() else torch.view_as_real(t) for t in tensors
    ]

    opts = AllreduceCoalescedOptions()
    opts.reduceOp = op
    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.allreduce_coalesced(tensors, opts)
    else:
        work = group.allreduce_coalesced(tensors, opts)

    if async_op:
        return work
    else:
        work.wait()


def reduce_multigpu(tensor_list,
                    dst,
                    op=ReduceOp.SUM,
                    group=None,
                    async_op=False,
                    dst_tensor=0,
                    country=None):
    """
    Reduces the tensor data on multiple GPUs across all machines. Each tensor
    in ``tensor_list`` should reside on a separate GPU

    Only the GPU of ``tensor_list[dst_tensor]`` on the process with rank ``dst``
    is going to receive the final result.

    Only nccl backend is currently supported
    tensors should only be GPU tensors

    Args:
        tensor_list (List[Tensor]): Input and output GPU tensors of the
            collective. The function operates in-place.
            You also need to make sure that ``len(tensor_list)`` is the same for
            all the distributed processes calling this function.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        dst_tensor (int, optional): Destination tensor rank within
                                    ``tensor_list``

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    assert country
    if country._rank_not_in_group(group):
        return

    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst
    opts.rootTensor = dst_tensor

    if group is None or group is country.WORLD:
        default_pg = country._get_default_group()
        work = default_pg.reduce(tensor_list, opts)
    else:
        group_dst_rank = country._get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.reduce(tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def reduce(tensor,
           dst,
           op=ReduceOp.SUM,
           group=None,
           async_op=False,
           country=None):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    assert country
    _check_single_tensor(tensor, "tensor")
    if country._rank_not_in_group(group):
        return

    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst

    if group is None or group is country.WORLD:
        default_pg = country._get_default_group()
        work = default_pg.reduce([tensor], opts)
    else:
        group_dst_rank = country._get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.reduce([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


def all_gather_multigpu(output_tensor_lists,
                        input_tensor_list,
                        group=None,
                        async_op=False,
                        country=None):
    """
    Gathers tensors from the whole group in a list.
    Each tensor in ``tensor_list`` should reside on a separate GPU

    Only nccl backend is currently supported
    tensors should only be GPU tensors

    Complex tensors are supported.

    Args:
        output_tensor_lists (List[List[Tensor]]): Output lists. It should
            contain correctly-sized tensors on each GPU to be used for output
            of the collective, e.g. ``output_tensor_lists[i]`` contains the
            all_gather result that resides on the GPU of
            ``input_tensor_list[i]``.

            Note that each element of ``output_tensor_lists`` has the size of
            ``world_size * len(input_tensor_list)``, since the function all
            gathers the result from every single GPU in the group. To interpret
            each element of ``output_tensor_lists[i]``, note that
            ``input_tensor_list[j]`` of rank k will be appear in
            ``output_tensor_lists[i][k * world_size + j]``

            Also note that ``len(output_tensor_lists)``, and the size of each
            element in ``output_tensor_lists`` (each element is a list,
            therefore ``len(output_tensor_lists[i])``) need to be the same
            for all the distributed processes calling this function.

        input_tensor_list (List[Tensor]): List of tensors(on different GPUs) to
            be broadcast from current process.
            Note that ``len(input_tensor_list)`` needs to be the same for
            all the distributed processes calling this function.

        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    assert country
    if country._rank_not_in_group(group):
        return

    output_tensor_lists = [[
        t if not t.is_complex() else torch.view_as_real(t) for t in l
    ] for l in output_tensor_lists]
    input_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t)
        for t in input_tensor_list
    ]

    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.allgather(output_tensor_lists, input_tensor_list)
    else:
        work = group.allgather(output_tensor_lists, input_tensor_list)

    if async_op:
        return work
    else:
        work.wait()


def all_gather_object(object_list, obj, group=None, country=None):
    """
    Gathers picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the object
    must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        object (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.all_gather_object(output, gather_objects[dist.get_rank(group)])
        >>> output
        ['foo', 12, {1: 2}]
    """
    assert country
    if country._rank_not_in_group(group):
        return

    input_tensor, local_size = _object_to_tensor(obj)
    group_backend = country.get_backend(group)
    is_nccl_backend = group_backend == Backend.NCCL
    current_device = torch.device("cpu")
    if is_nccl_backend:
        # See note about using torch.cuda.current_device() here in docstring.
        # We cannot simply use my_rank since rank == device is not necessarily
        # true.
        current_device = torch.device('cuda', torch.cuda.current_device())
        input_tensor = input_tensor.to(current_device)
        local_size = local_size.to(current_device)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = country.get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size,
                                      dtype=torch.long,
                                      device=current_device)
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    all_gather(object_size_list, local_size, group=group, country=country)
    max_object_size = int(max(object_size_list).item())  # type: ignore
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(max_object_size * group_size,
                                          dtype=torch.uint8,
                                          device=current_device)
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i:max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    all_gather(output_tensors, input_tensor, group=group, country=country)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.ByteTensor)  # type:ignore[call-overload]
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


def gather_object(obj,
                  object_gather_list=None,
                  dst=0,
                  group=None,
                  country=None) -> None:
    """
    Gathers picklable objects from the whole group in a single process.
    Similar to :func:`gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Args:
        obj (Any): Input object. Must be picklable.
        object_gather_list (list[Any]): Output list. On the ``dst`` rank, it
            should be correctly sized as the size of the group for this
            collective and will contain the output. Must be ``None`` on non-dst
            ranks. (default is ``None``)
        dst (int, optional): Destination rank. (default is 0)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. On the ``dst`` rank, ``object_gather_list`` will contain the
        output of the collective.

    .. note:: Note that this API is not supported when using the NCCL backend.

    .. warning::
        :func:`gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.gather_object(
                gather_objects[dist.get_rank(group)],
                output if dist.get_rank(group) == 0 else None,
                dst=0
            )
        >>> # On rank 0
        >>> output
        ['foo', 12, {1: 2}]
    """
    assert country
    if country._rank_not_in_group(group):
        return

    # Ensure object_gather_list is specified appropriately.
    my_rank = country.get_rank()
    country._validate_output_list_for_rank(my_rank, dst, object_gather_list)
    input_tensor, local_size = _object_to_tensor(obj)
    group_backend = country.get_backend(group)
    current_device = torch.device("cpu")
    is_nccl_backend = group_backend == Backend.NCCL
    if is_nccl_backend:
        current_device = torch.device('cuda', torch.cuda.current_device())
        input_tensor = input_tensor.to(current_device)
        local_size = local_size.to(current_device)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = country.get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size,
                                      dtype=torch.long,
                                      device=current_device)
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]

    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    all_gather(object_size_list, local_size, group=group, country=country)

    max_object_size = int(max(object_size_list).item())  # type: ignore
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_rank == dst:
        coalesced_output_tensor = torch.empty(max_object_size * group_size,
                                              dtype=torch.uint8,
                                              device=current_device)
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i:max_object_size *
                                    (i + 1)] for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    gather(
        input_tensor,
        gather_list=output_tensors if my_rank == dst else None,  # type: ignore
        dst=dst,
        group=group,
        async_op=False,
        country=country)

    if my_rank != dst:
        return
    for i, tensor in enumerate(output_tensors):  # type: ignore
        # type: ignore[call-overload]
        tensor = tensor.type(torch.ByteTensor)  # type: ignore
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size)


def broadcast_object_list(object_list, src=0, group=None, country=None):
    """
    Broadcasts picklable objects in ``object_list`` to the whole group. Similar
    to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank(group) == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> dist.broadcast_object_list(objects, src=0)
        >>> broadcast_objects
        ['foo', 12, {1: 2}]
    """
    assert country
    if country._rank_not_in_group(group):
        return

    my_rank = country.get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.LongTensor(len(object_list))

    group_backend = country.get_backend(group)
    is_nccl_backend = group_backend == Backend.NCCL
    current_device = torch.device("cpu")
    if is_nccl_backend:
        # See note about using torch.cuda.current_device() here in docstring.
        # We cannot simply use my_rank since rank == device is not necessarily
        # true.
        current_device = torch.device('cuda', torch.cuda.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    broadcast(object_sizes_tensor, src=src, group=group, country=country)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)  # type: ignore
    else:
        object_tensor = torch.ByteTensor(torch.sum(object_sizes_tensor).item())

    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)
    broadcast(object_tensor, src=src, group=group, country=country)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset:offset + obj_size]
            # type: ignore[call-overload]
            obj_view = obj_view.type(torch.ByteTensor)  # type: ignore
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)


def scatter_object_list(scatter_object_output_list,
                        scatter_object_input_list,
                        src=0,
                        group=None,
                        country=None):
    """
    Scatters picklable objects in ``scatter_object_input_list`` to the whole
    group. Similar to :func:`scatter`, but Python objects can be passed in. On
    each rank, the scattered object will be stored as the first element of
    ``scatter_object_output_list``. Note that all objects in
    ``scatter_object_input_list`` must be picklable in order to be scattered.

    Args:
        scatter_object_output_list (List[Any]): Non-empty list whose first
            element will store the object scattered to this rank.
        scatter_object_input_list (List[Any]): List of input objects to scatter.
            Each object must be picklable. Only objects on the ``src`` rank will
            be scattered, and the argument can be ``None`` for non-src ranks.
        src (int): Source rank from which to scatter
            ``scatter_object_input_list``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        ``None``. If rank is part of the group, ``scatter_object_output_list``
        will have its first element set to the scattered object for this rank.

    .. note:: Note that this API differs slightly from the scatter collective
        since it does not provide an ``async_op`` handle and thus will be a
        blocking call.

    .. warning::
        :func:`scatter_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank(group) == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     # Can be any list on non-src ranks, elements are not used.
        >>>     objects = [None, None, None]
        >>> output_list = [None]
        >>> dist.scatter_object_list(output_list, objects, src=0)
        >>> # Rank i gets objects[i]. For example, on rank 2:
        >>> output_list
        [{1: 2}]
    """
    assert country
    if country._rank_not_in_group(group):
        return

    if (not isinstance(scatter_object_output_list, list)
            or len(scatter_object_output_list) < 1):
        raise RuntimeError(
            "Expected argument scatter_object_output_list to be a list of size at least 1."
        )

    my_rank = country.get_rank()
    if my_rank == src:
        tensor_list, tensor_sizes = zip(
            *[_object_to_tensor(obj) for obj in scatter_object_input_list])
        tensor_list, tensor_sizes = list(tensor_list), list(tensor_sizes)

    obj_tensor_size = torch.LongTensor([0])
    # Src rank broadcasts the maximum tensor size. This is because all ranks are
    # expected to call into scatter() with equal-sized tensors.
    if my_rank == src:
        max_tensor_size = max(tensor_sizes)  # type: ignore
        for tensor in tensor_list:  # type: ignore
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.LongTensor([0])
    broadcast(max_tensor_size, src=src, group=group, country=country)

    # Scatter actual serialized objects
    output_tensor = torch.ByteTensor(max_tensor_size.item())
    scatter(
        output_tensor,
        scatter_list=None if my_rank != src else tensor_list,  # type: ignore
        src=src,
        group=group,
    )

    # Scatter per-object sizes to trim tensors when deserializing back to object
    scatter(
        obj_tensor_size,
        scatter_list=None if my_rank != src else tensor_sizes,  # type: ignore
        src=src,
        group=group,
    )

    # Deserialize back to object
    scatter_object_output_list[0] = _tensor_to_object(output_tensor,
                                                      obj_tensor_size)


def all_gather(tensor_list, tensor, group=None, async_op=False, country=None):
    """
    Gathers tensors from the whole group in a list.

    Complex tensors are supported.

    Args:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [torch.zero(2, dtype=torch.int64) for _ in range(2)]
        >>> tensor_list
        [tensor([0, 0]), tensor([0, 0])] # Rank 0 and 1
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1, 2]), tensor([3, 4])] # Rank 0
        [tensor([1, 2]), tensor([3, 4])] # Rank 1

        >>> # All tensors below are of torch.cfloat dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [torch.zero(2, dtype=torch.cfloat) for _ in range(2)]
        >>> tensor_list
        [tensor([0.+0.j, 0.+0.j]), tensor([0.+0.j, 0.+0.j])] # Rank 0 and 1
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j]) # Rank 0
        tensor([3.+3.j, 4.+4.j]) # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 0
        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 1

    """
    assert country
    _check_tensor_list(tensor_list, "tensor_list")
    _check_single_tensor(tensor, "tensor")
    if country._rank_not_in_group(group):
        return

    tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in tensor_list
    ]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)

    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.allgather([tensor_list], [tensor])
    else:
        work = group.allgather([tensor_list], [tensor])

    if async_op:
        return work
    else:
        work.wait()


def all_gather_coalesced(output_tensor_lists,
                         input_tensor_list,
                         group=None,
                         async_op=False,
                         country=None):
    """
    Gathers input tensors from the whole group in a list in a coalesced manner.

    Complex tensors are supported.

    Args:
        output_tensor_lists (list[list[Tensor]]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor_list (list[Tensor]): Tensors to be broadcast from
            current process. At least one tensor has to be non empty.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Example:
        we have 2 process groups, 2 ranks.
        rank 0 passes:
            input_tensor_list = [[[1, 1], [1, 1]], [2], [3, 3]]
            output_tensor_lists =
            [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        rank 1 passes:
            input_tensor_list = [[[3, 3], [3, 3]], [5], [1, 1]]
            output_tensor_lists =
            [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        both rank 0 and 1 get:
            output_tensor_lists =
            [[[1, 1], [1, 1]], [2], [3, 3]],
                [[3, 3], [3, 3]], [5], [1, 1]]].

    WARNING: at this time individual shape checking is not implemented across nodes.
    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the
    all_gather_coalesced operation will proceed without complaint and return
    erroneous outputs. This lack of shape checking results in significant
    performance improvements but users of this function should take extra care
    to ensure that each node passes in tensors whose shapes match across nodes.
    """
    assert country
    # We only check basic compatibility with C++ params here, C++ code will
    # do shape and type checking.
    if country._rank_not_in_group(group):
        return
    _check_tensor_list(input_tensor_list, "tensor_list")
    if not isinstance(output_tensor_lists, list):
        raise RuntimeError("Invalid function argument: "
                           "output_tensor_lists should be a list")
    for output_tensor_list in output_tensor_lists:
        _check_tensor_list(output_tensor_list, "output_tensor_lists")

    output_tensor_lists = [[
        t if not t.is_complex() else torch.view_as_real(t) for t in l
    ] for l in output_tensor_lists]
    input_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t)
        for t in input_tensor_list
    ]

    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.allgather_coalesced(output_tensor_lists,
                                              input_tensor_list)
    else:
        work = group.allgather_coalesced(output_tensor_lists,
                                         input_tensor_list)

    if async_op:
        return work
    else:
        work.wait()


def gather(tensor,
           gather_list=None,
           dst=0,
           group=None,
           async_op=False,
           country=None):
    """
    Gathers a list of tensors in a single process.

    Args:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    assert country
    _check_single_tensor(tensor, "tensor")

    # Parameter ``gather_list`` may be left unspecified on non-dst ranks.
    if gather_list:
        _check_tensor_list(gather_list, "gather_list")
    else:
        gather_list = []

    if country._rank_not_in_group(group):
        return

    my_rank = country.get_rank()
    country._validate_output_list_for_rank(my_rank, dst, gather_list)
    output_tensors = [gather_list] if dst == my_rank else []
    input_tensors = [tensor]

    opts = GatherOptions()
    opts.rootRank = dst

    if group is None or group is country.WORLD:
        default_pg = country._get_default_group()
        work = default_pg.gather(output_tensors, input_tensors, opts)
    else:
        # https://github.com/pytorch/pytorch/issues/60356
        group_dst_rank = country._get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.gather(output_tensors, input_tensors, opts)

    if async_op:
        return work
    else:
        work.wait()


def scatter(tensor,
            scatter_list=None,
            src=0,
            group=None,
            async_op=False,
            country=None):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Args:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank)
        src (int): Source rank (default is 0)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    assert country
    _check_single_tensor(tensor, "tensor")

    # Parameter ``scatter_list`` may be left unspecified on non-src ranks.
    if scatter_list:
        _check_tensor_list(scatter_list, "scatter_list")
    else:
        scatter_list = []

    if country._rank_not_in_group(group):
        return

    my_rank = country.get_rank()
    if src == my_rank:
        if not scatter_list:
            raise ValueError("Argument ``scatter_list`` must be specified "
                             "on source rank.")
        input_tensors = [scatter_list]
        output_tensors = [tensor]
    else:
        if scatter_list:
            raise ValueError("Argument ``scatter_list`` must NOT be specified "
                             "on non-source ranks.")
        input_tensors = []
        output_tensors = [tensor]

    opts = ScatterOptions()
    opts.rootRank = src

    if group is None or group is country.WORLD:
        default_pg = country._get_default_group()
        work = default_pg.scatter(output_tensors, input_tensors, opts)
    else:
        group_src_rank = country._get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.scatter(output_tensors, input_tensors, opts)

    if async_op:
        return work
    else:
        work.wait()


def reduce_scatter_multigpu(output_tensor_list,
                            input_tensor_lists,
                            op=ReduceOp.SUM,
                            group=None,
                            async_op=False,
                            country=None):
    """
    Reduce and scatter a list of tensors to the whole group.  Only nccl backend
    is currently supported.

    Each tensor in ``output_tensor_list`` should reside on a separate GPU, as
    should each list of tensors in ``input_tensor_lists``.

    Args:
        output_tensor_list (List[Tensor]): Output tensors (on different GPUs)
            to receive the result of the operation.

            Note that ``len(output_tensor_list)`` needs to be the same for all
            the distributed processes calling this function.

        input_tensor_lists (List[List[Tensor]]): Input lists.  It should
            contain correctly-sized tensors on each GPU to be used for input of
            the collective, e.g. ``input_tensor_lists[i]`` contains the
            reduce_scatter input that resides on the GPU of
            ``output_tensor_list[i]``.

            Note that each element of ``input_tensor_lists`` has the size of
            ``world_size * len(output_tensor_list)``, since the function
            scatters the result from every single GPU in the group.  To
            interpret each element of ``input_tensor_lists[i]``, note that
            ``output_tensor_list[j]`` of rank k receives the reduce-scattered
            result from ``input_tensor_lists[i][k * world_size + j]``

            Also note that ``len(input_tensor_lists)``, and the size of each
            element in ``input_tensor_lists`` (each element is a list,
            therefore ``len(input_tensor_lists[i])``) need to be the same for
            all the distributed processes calling this function.

        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert country
    if country._rank_not_in_group(group):
        return

    opts = ReduceScatterOptions()
    opts.reduceOp = op

    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.reduce_scatter(output_tensor_list,
                                         input_tensor_lists, opts)
    else:
        work = group.reduce_scatter(output_tensor_list, input_tensor_lists,
                                    opts)

    if async_op:
        return work
    else:
        work.wait()


def reduce_scatter(output,
                   input_list,
                   op=ReduceOp.SUM,
                   group=None,
                   async_op=False,
                   country=None):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert country
    _check_single_tensor(output, "output")
    _check_tensor_list(input_list, "input_list")
    if country._rank_not_in_group(group):
        return

    opts = ReduceScatterOptions()
    opts.reduceOp = op

    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.reduce_scatter([output], [input_list], opts)
    else:
        work = group.reduce_scatter([output], [input_list], opts)

    if async_op:
        return work
    else:
        work.wait()


def all_to_all_single(output,
                      input,
                      output_split_sizes=None,
                      input_split_sizes=None,
                      group=None,
                      async_op=False,
                      country=None):
    """
    Each process splits input tensor and then scatters the split list
    to all processes in a group. Then concatenate the received tensors from all
    the processes in the group and return single output tensor.

    Args:
        output (Tensor): Gathered cancatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all_single` is experimental and subject to change.

    Examples:
        >>> input = torch.arange(4) + rank * 4
        >>> input
        tensor([0, 1, 2, 3])     # Rank 0
        tensor([4, 5, 6, 7])     # Rank 1
        tensor([8, 9, 10, 11])   # Rank 2
        tensor([12, 13, 14, 15]) # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([0, 4, 8, 12])    # Rank 0
        tensor([1, 5, 9, 13])    # Rank 1
        tensor([2, 6, 10, 14])   # Rank 2
        tensor([3, 7, 11, 15])   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = list(input.chunk(world_size))
        >>> gather_list  = list(output.chunk(world_size))
        >>> for i in range(world_size):
        >>>   dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)

        >>> # Another example with uneven split
        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> output = ...
        >>> dist.all_to_all_single(output, input, output_splits, input_splits)
        >>> output
        tensor([ 0,  1, 10, 11, 12, 20, 21, 30, 31])                     # Rank 0
        tensor([ 2,  3, 13, 14, 22, 32, 33])                             # Rank 1
        tensor([ 4, 15, 16, 23, 34, 35])                                 # Rank 2
        tensor([ 5, 17, 18, 24, 36])                                     # Rank 3
    """
    assert country
    if country._rank_not_in_group(group):
        return

    opts = AllToAllOptions()
    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    output_split_sizes = [] if output_split_sizes is None else output_split_sizes
    input_split_sizes = [] if input_split_sizes is None else input_split_sizes

    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.alltoall_base(output, input, output_split_sizes,
                                        input_split_sizes, opts)
    else:
        work = group.alltoall_base(output, input, output_split_sizes,
                                   input_split_sizes, opts)

    if async_op:
        return work
    else:
        work.wait()


def all_to_all(output_tensor_list,
               input_tensor_list,
               group=None,
               async_op=False,
               country=None):
    """
    Each process scatters list of input tensors to all processes in a group and
    return gathered list of tensors in output list.

    Args:
        output_tensor_list (list[Tensor]): List of tensors to be gathered one
            per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all` is experimental and subject to change.

    Examples:
        >>> input = torch.arange(4) + rank * 4
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
        [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
        [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
        [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0
        [tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1
        [tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2
        [tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = input
        >>> gather_list  = output
        >>> for i in range(world_size):
        >>>   dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)

        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> input = list(input.split(input_splits))
        >>> input
        [tensor([0, 1]), tensor([2, 3]), tensor([4]), tensor([5])]                   # Rank 0
        [tensor([10, 11, 12]), tensor([13, 14]), tensor([15, 16]), tensor([17, 18])] # Rank 1
        [tensor([20, 21]), tensor([22]), tensor([23]), tensor([24])]                 # Rank 2
        [tensor([30, 31]), tensor([32, 33]), tensor([34, 35]), tensor([36])]         # Rank 3
        >>> output = ...
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0, 1]), tensor([10, 11, 12]), tensor([20, 21]), tensor([30, 31])]   # Rank 0
        [tensor([2, 3]), tensor([13, 14]), tensor([22]), tensor([32, 33])]           # Rank 1
        [tensor([4]), tensor([15, 16]), tensor([23]), tensor([34, 35])]              # Rank 2
        [tensor([5]), tensor([17, 18]), tensor([24]), tensor([36])]                  # Rank 3
    """
    assert country
    if country._rank_not_in_group(group):
        return

    opts = AllToAllOptions()
    _check_tensor_list(output_tensor_list, "output_tensor_list")
    _check_tensor_list(input_tensor_list, "input_tensor_list")

    if group is None:
        default_pg = country._get_default_group()
        work = default_pg.alltoall(output_tensor_list, input_tensor_list, opts)
    else:
        work = group.alltoall(output_tensor_list, input_tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def batch_isend_irecv(p2p_op_list, country=None):
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
    assert country
    country._check_p2p_op_list(p2p_op_list)
    backend = country.get_backend(p2p_op_list[0].group)
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
