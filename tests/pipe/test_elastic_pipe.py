import torch
from openfed.pipe.elastic_pipe import ElasticPipe


def test_elastic_pipe():
    ElasticPipe([torch.nn.Parameter(torch.randn(1))])
