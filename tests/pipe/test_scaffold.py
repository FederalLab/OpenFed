import torch
from openfed.pipe.scaffold_pipe import ScaffoldPipe


def test_scaffold_pipe():
    ScaffoldPipe([torch.nn.Parameter(torch.randn(1))])
