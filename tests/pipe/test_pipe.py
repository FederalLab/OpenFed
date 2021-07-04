import torch
from openfed.pipe.pipe import Pipe


def test_pipe():
    Pipe([torch.nn.Parameter(torch.randn(1))], dict())
