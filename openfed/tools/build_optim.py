import openfed.optim
import openfed.core
import openfed.container
import torch.optim as optim


def build_fedsgd(parameters, lr, role, **kwargs):
    """Build fedsgd, return optimizer and aggregator (for leader).
    """
    parameters = list(parameters)
    optimizer = optim.SGD(
        parameters, lr=lr, **kwargs) if 'optimizer' not in kwargs else kwargs['optimizer']
    fed_optimizer = openfed.optim.build_fed_optim(optimizer)
    if openfed.core.is_leader(role):
        aggregator = openfed.container.AverageAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedavg(parameters, lr, role, **kwargs):
    """Build fedavg, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    optimizer = optim.SGD(parameters, lr=lr, **kwargs)
    fed_optimizer = openfed.optim.build_fed_optim(
        optimizer) if 'optimizer' not in kwargs else kwargs['optimizer']
    if openfed.core.is_leader(role):
        aggregator = openfed.container.NaiveAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedela(parameters, lr, role, **kwargs):
    """Build fedela, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    optimizer = optim.SGD(
        parameters, lr=lr, **kwargs) if 'optimizer' not in kwargs else kwargs['optimizer']
    penalizer = openfed.optim.ElasticPenalizer(role)
    fed_optimizer = openfed.optim.build_fed_optim(optimizer, penalizer)

    if openfed.core.is_leader(role):
        aggregator = openfed.container.ElasticAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedprox(parameters, lr, role, **kwargs):
    """Build fedprox, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    optimizer = optim.SGD(
        parameters, lr=lr, **kwargs) if 'optimizer' not in kwargs else kwargs['optimizer']
    penalizer = openfed.optim.ProxPenalizer(role)
    fed_optimizer = openfed.optim.build_fed_optim(optimizer, penalizer)

    if openfed.core.is_leader(role):
        aggregator = openfed.container.NaiveAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedscaffold(parameters, lr, role, **kwargs):
    """Build fedscaffold, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    optimizer = optim.SGD(
        parameters, lr=lr, **kwargs) if 'optimizer' not in kwargs else kwargs['optimizer']
    penalizer = openfed.optim.ScaffoldPenalizer(
        role, pack_set=['c_para'], unpack_set=['c_para'])
    fed_optimizer = openfed.optim.build_fed_optim(optimizer, penalizer)

    if openfed.core.is_leader(role):
        aggregator = openfed.container.NaiveAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_optim(name, *args, **kwargs):
    if name == 'fedavg':
        return build_fedavg(*args, **kwargs)
    elif name == 'fedsgd':
        return build_fedsgd(*args, **kwargs)
    elif name == 'fedela':
        return build_fedela(*args, **kwargs)
    elif name == 'fedprox':
        return build_fedprox(*args, **kwargs)
    elif name == 'fedscaffold':
        return build_fedscaffold(*args, **kwargs)
    else:
        raise ValueError('Unknown federated optimizer')
