from openfed.common.vars import *


def test_vars():
    DAL.set_dal()
    assert DAL.is_dal

    DAL.unset_dal()
    assert not DAL.is_dal

    print(DAL)

    ASYNC_OP.set_async_op()
    assert ASYNC_OP.is_async_op

    ASYNC_OP.unset_async_op()
    assert not ASYNC_OP.is_async_op
