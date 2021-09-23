from multiprocessing import Process


def aggregator():
    # build a topology first
    import openfed
    import openfed.topo as topo

    aggregator = topo.Node('aggregator', openfed.default_tcp_address)
    alpha = topo.Node('alpha', openfed.empty_address)
    beta = topo.Node('beta', openfed.empty_address)

    topology = topo.Topology()
    topology.add_node_list([aggregator, alpha, beta])
    topology.add_edge(alpha, aggregator)
    topology.add_edge(beta, aggregator)

    # analysis topology to get federated group props
    federated_group_props = topo.analysis(topology, aggregator)
    assert len(federated_group_props) == 1
    federated_group_prop = federated_group_props[0]

    # build network
    import torch.nn as nn
    network = nn.Linear(10, 1)

    # build maintainer
    from openfed.core import Maintainer
    maintainer = Maintainer(federated_group_prop,
                            network.state_dict(keep_vars=True))

    with maintainer:
        openfed.functional.device_alignment()
        openfed.functional.count_step(2)

    maintainer.step()


def collaborator_alpha():
    # build a topology first
    import openfed
    import openfed.topo as topo

    aggregator = topo.Node('aggregator', openfed.default_tcp_address)
    alpha = topo.Node('alpha', openfed.empty_address)
    beta = topo.Node('beta', openfed.empty_address)

    topology = topo.Topology()
    topology.add_node_list([aggregator, alpha, beta])
    topology.add_edge(alpha, aggregator)
    topology.add_edge(beta, aggregator)

    # analysis topology to get federated group props
    federated_group_props = topo.analysis(topology, alpha)
    assert len(federated_group_props) == 1
    federated_group_prop = federated_group_props[0]

    # build network
    import torch.nn as nn
    network = nn.Linear(10, 1)

    # build maintainer
    from openfed.core import Maintainer
    maintainer = Maintainer(federated_group_prop,
                            network.state_dict(keep_vars=True))

    with maintainer:
        openfed.functional.device_alignment()

    maintainer.step(upload=False)
    maintainer.package()
    maintainer.step(download=False)


def collaborator_beta():
    # build a topology first
    import openfed
    import openfed.topo as topo

    aggregator = topo.Node('aggregator', openfed.default_tcp_address)
    alpha = topo.Node('alpha', openfed.empty_address)
    beta = topo.Node('beta', openfed.empty_address)

    topology = topo.Topology()
    topology.add_node_list([aggregator, alpha, beta])
    topology.add_edge(alpha, aggregator)
    topology.add_edge(beta, aggregator)

    # analysis topology to get federated group props
    federated_group_props = topo.analysis(topology, beta)
    assert len(federated_group_props) == 1
    federated_group_prop = federated_group_props[0]

    # build network
    import torch.nn as nn
    network = nn.Linear(10, 1)

    # build maintainer
    from openfed.core import Maintainer
    maintainer = Maintainer(federated_group_prop,
                            network.state_dict(keep_vars=True))

    with maintainer:
        openfed.functional.device_alignment()

    maintainer.step(upload=False)
    maintainer.package()
    maintainer.step(download=False)


def test_maintainer():
    aggregator_p = Process(target=aggregator)
    collaborator_alpha_p = Process(target=collaborator_alpha)
    collaborator_beta_p = Process(target=collaborator_beta)

    aggregator_p.start()
    collaborator_alpha_p.start()
    collaborator_beta_p.start()

    aggregator_p.join()
    collaborator_alpha_p.join()
    collaborator_beta_p.join()
