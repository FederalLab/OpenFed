import openfed.core
import openfed.optim as fed_optim
import torch.optim as optim


def build_fedsgd(parameters, lr, role, **kwargs):
    """Build fedsgd, return optimizer and aggregator (for leader).
    """
    reducer = kwargs.pop('reducer') if 'reducer' in kwargs else None

    parameters = list(parameters)
    optimizer = optim.SGD(
        parameters, lr=lr, **kwargs) if 'optimizer' not in kwargs else kwargs['optimizer']
    fed_optimizer = fed_optim.build_fed_optim(optimizer)
    if openfed.core.is_leader(role):
        agg = fed_optim.AverageOp(parameters)
        aggregator = fed_optim.build_aggregator(agg, reducer)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedavg(parameters, lr, role, **kwargs):
    """Build fedavg, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    parameters = list(parameters)

    reducer = kwargs.pop('reducer') if 'reducer' in kwargs else None
    
    optimizer = optim.SGD(parameters, lr=lr, **kwargs) if 'optimizer' not in kwargs else kwargs['optimizer']
    
    fed_optimizer = fed_optim.build_fed_optim(optimizer) 
    if openfed.core.is_leader(role):
        agg = fed_optim.NaiveOp(parameters)
        aggregator = fed_optim.build_aggregator(agg, reducer)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedela(parameters, lr, role, **kwargs):
    """Build fedela, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    parameters = list(parameters)
    reducer = kwargs.pop('reducer') if 'reducer' in kwargs else None
    penal_cfg = kwargs.pop('penal_cfg') if 'penal_cfg' in kwargs else dict()
    optimizer = optim.SGD(
        parameters, lr=lr, **kwargs) if 'optimizer' not in kwargs else kwargs['optimizer']
    
    penalizer = fed_optim.ElasticPenalizer(role, **penal_cfg)
    fed_optimizer = fed_optim.build_fed_optim(optimizer, penalizer)

    if openfed.core.is_leader(role):
        agg = fed_optim.ElasticOp(parameters)
        aggregator = fed_optim.build_aggregator(agg, reducer)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedprox(parameters, lr, role, **kwargs):
    """Build fedprox, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    parameters = list(parameters)
    reducer = kwargs.pop('reducer') if 'reducer' in kwargs else None
    penal_cfg = kwargs.pop('penal_cfg') if 'penal_cfg' in kwargs else dict()

    optimizer = optim.SGD(
        parameters, lr=lr, **kwargs) if 'optimizer' not in kwargs else kwargs['optimizer']
    
    penalizer = fed_optim.ProxPenalizer(role, **penal_cfg)
    fed_optimizer = fed_optim.build_fed_optim(optimizer, penalizer)

    if openfed.core.is_leader(role):
        agg = fed_optim.NaiveOp(parameters)
        aggregator = fed_optim.build_aggregator(agg, reducer)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedscaffold(parameters, lr, role, **kwargs):
    """Build fedscaffold, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    parameters = list(parameters)
    reducer = kwargs.pop('reducer') if 'reducer' in kwargs else None
    penal_cfg = kwargs.pop('penal_cfg') if 'penal_cfg' in kwargs else dict(pack_set=['c_para'], unpack_set=['c_para'])
    optimizer = optim.SGD(
        parameters, lr=lr, **kwargs) if 'optimizer' not in kwargs else kwargs['optimizer']
    
    penalizer = fed_optim.ScaffoldPenalizer(role, **penal_cfg)
    fed_optimizer = fed_optim.build_fed_optim(optimizer, penalizer)

    if openfed.core.is_leader(role):
        agg = fed_optim.NaiveOp(parameters)
        aggregator = fed_optim.build_aggregator( agg, reducer)
    else:
        aggregator = None

    return fed_optimizer, aggregator

builder = dict(
    fedavg=build_fedavg,
    fedsgd=build_fedsgd,
    fedela=build_fedela,
    fedprox=build_fedprox,
    fedscaffold=build_fedscaffold,
)

def build_optim(name, *args, **kwargs):
    """Returns a optimizer and aggregator (for leader).
    """
    if name not in builder:
        raise KeyError(f"Not implemented fed optimizer: {name}")
    return builder[name](*args, **kwargs)
