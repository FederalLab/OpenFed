import torch
from openfed.container.agg.agg import Agg


def test_agg():
    agg = Agg([torch.nn.Parameter(torch.randn(1))], dict(), list(), list())

    # save
    torch.save(agg.state_dict(), '/tmp/openfed.test')

    # load
    agg.load_state_dict(torch.load('/tmp/openfed.test'))
