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


class _DYNAMIC_ADDRESS_LOADING(Vars):
    def __init__(self):
        self.flag = True

    def set_dynamic_address_loading(self):
        self.flag = True

    def unset_dynamic_address_loading(self):
        self.flag = False

    @property
    def is_dynamic_address_loading(self) -> bool:
        return self.flag

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="DYNAMIC_ADDRESS_LOADING",
            description="If dynamic_address_loading is enabled, a thread will be created to maintain new connection."
                        "Otherwise, OpenFed will get stuck until all address are correctly jointed."
                        "Call set_dynamic_address_loading() to set dynamic_address_loading, unset_dynamic_address_loading to disable it."
        )


DYNAMIC_ADDRESS_LOADING = _DYNAMIC_ADDRESS_LOADING()


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
