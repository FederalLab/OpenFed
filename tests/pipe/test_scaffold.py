import torch
from openfed.pipe.scaffold import Scaffold


def test_scaffold():
    Scaffold([torch.nn.Parameter(torch.randn(1))])
