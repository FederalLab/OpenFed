import torch
from openfed.common.package import Package
from torch.nn import Linear
from torch.optim import SGD


class MyPackage(Package):
    def pack(self, key, rdict):
        return None

    def unpack(self, key, rdict):
        return rdict


def test_package():
    my_package = MyPackage()

    net = Linear(1, 1)
    optimizer = SGD(net.parameters(), lr=1.0, momentum=0.1)
    net(torch.randn(1, 1, 1)).sum().backward()
    optimizer.step()

    my_package.pack_state(optimizer, keys="momentum_buffer")

    my_package.unpack_state(optimizer, keys="momentum_buffer")
