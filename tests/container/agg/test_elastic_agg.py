from openfed.container.agg.elastic_agg import ElasticAgg
import torch


def test_average_agg():
    agg = ElasticAgg([torch.nn.Parameter(torch.randn(1))])

    # save
    torch.save(agg.state_dict(), '/tmp/openfed.test')

    # load
    agg.load_state_dict(torch.load('/tmp/openfed.test'))
