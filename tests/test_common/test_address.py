# @Author            : FederalLab
# @Date              : 2021-09-25 16:56:33
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:56:33
# Copyright (c) FederalLab. All rights reserved.
from openfed import (Address, default_file_address, default_tcp_address,
                     empty_address)


def test_backend():
    Address('gloo', 'null')
    Address('mpi', 'null')
    address = Address('null', 'null')

    data = address.serialize()
    address.unserialize(data)


def test_tcp_address():
    tcp_address = Address(
        'gloo', init_method='tcp://localhost:1994', rank=1, world_size=2)

    assert tcp_address == default_tcp_address


def test_file_address():
    file_address = Address(
        'gloo',
        init_method='file:///tmp/openfed.sharedfile',
        rank=1,
        world_size=2)

    assert file_address == default_file_address


def test_empty_address():
    empty_address_tmp = Address('null', 'null')

    assert empty_address_tmp == empty_address
