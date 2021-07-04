import torch
from openfed.pipe.prox_pipe import ProxPipe


def test_prox_pipe():
    ProxPipe([torch.nn.Parameter(torch.randn(1))])
