from torch._six import string_classes


class Backend(object):
    """
    An enum-like class of available backends: GLOO, NCCL, MPI, and other registered
    backends.

    The values of this class are lowercase strings, e.g., ``"gloo"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend("GLOO")`` returns ``"gloo"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
            initial value of some fields. Users should neither use it directly
            nor assume its existence.
    """
    UNDEFINED = "undefined"
    GLOO = "gloo"
    NCCL = "nccl"
    MPI = "mpi"
    TCP = "tcp"

    def __new__(cls, name: str):
        if not isinstance(name, string_classes):
            raise ValueError(
                "Backend name must be a string, but got: {}".format(name))
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)

        if value == Backend.TCP:
            raise ValueError("TCP backend has been deprecated. Please use "
                             "Gloo or MPI backend for collective operations "
                             "on CPU tensors.")
        elif value == Backend.UNDEFINED:
            raise ValueError("Invalid backend: '{}'".format(name))
        elif value != Backend.GLOO and value != Backend.NCCL and value != Backend.MPI:
            value = name
        return value

    @classmethod
    def register_backend(cls, name, func):
        """
        Registers a new backend.

        This class method is used by 3rd party cpp extension to register new backend.

        Args:
            name (str): Backend name matching with the one in `init_process_group()`.
            func (function): Function handler that instantiates the backend.
                            The function should be implemented in the backend cpp extension
                            and takes four arguments, including prefix_store, rank,
                            world_size, and timeout.

        .. note:: This support of 3rd party backend is experimental and subject to change.

        """
        setattr(Backend, name.upper(), func)
