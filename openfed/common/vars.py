from openfed.utils import openfed_class_fmt


class Vars(object):
    flag: bool

    def __init__(self):
        self.flag = False

    def set(self, flag: bool):
        self.flag = flag

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Vars",
            description="Base class for global variables."
        )


class _DAL(Vars):
    def __init__(self):
        self.flag = True

    def set_dal(self):
        self.flag = True

    def unset_dal(self):
        self.flag = False

    @property
    def is_dal(self) -> bool:
        return self.flag

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="DAL",
            description="If dal is enabled, a thread will be created to maintain new connection."
                        "Otherwise, OpenFed will get stuck until all address are correctly jointed."
                        "Call set_dal() to set dal, unset_dal to disable it."
        )


DAL = _DAL()


class _ASYNC_OP(Vars):
    def __init__(self):
        self.flag = True

    def set_async_op(self):
        self.flag = True

    def unset_async_op(self):
        self.flag = False

    @property
    def is_async_op(self) -> bool:
        return self.flag

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="ASYNC_OP",
            description="If True, the download and upload operation will return an handler."
                        "Otherwise, it will be blocked until finish."
        )


ASYNC_OP = _ASYNC_OP()
