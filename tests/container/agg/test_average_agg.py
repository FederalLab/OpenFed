from openfed.container.agg.average_agg import AverageAgg
import torch


def test_average_agg():
    agg = AverageAgg([torch.nn.Parameter(torch.randn(1))])

    # save
    torch.save(agg.state_dict(), '/tmp/openfed.test')

    # load
    agg.load_state_dict(torch.load('/tmp/openfed.test'))
