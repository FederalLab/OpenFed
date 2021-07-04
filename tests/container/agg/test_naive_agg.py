import torch
from openfed.container.agg.naive_agg import NaiveAgg


def test_average_agg():
    agg = NaiveAgg([torch.nn.Parameter(torch.randn(1))])

    # save
    torch.save(agg.state_dict(), '/tmp/openfed.test')

    # load
    agg.load_state_dict(torch.load('/tmp/openfed.test'))
