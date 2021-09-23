import openfed
import openfed.topo as topo


def test_node():
    alpha = topo.Node('alpha', address=openfed.empty_address)

    assert alpha.nick_name == 'alpha'
    assert alpha.address == openfed.empty_address

    beta = topo.Node('beta', address=openfed.empty_address)

    assert alpha != beta

    alpha_copy = topo.Node('alpha', address=openfed.empty_address)

    assert alpha_copy == alpha

    print(alpha)


def test_edge():
    alpha = topo.Node('alpha', address=openfed.empty_address)
    beta = topo.Node('beta', address=openfed.empty_address)

    edge = topo.Edge(alpha, beta)
    r_edge = topo.Edge(beta, alpha)

    assert edge != r_edge

    edge_copy = topo.Edge(alpha, beta)

    assert edge_copy == edge

    print(edge)


def test_topology():
    alpha = topo.Node('alpha', address=openfed.empty_address)

    topology = topo.Topology()

    topology.add_node(alpha)
    topology.add_node('beta', openfed.empty_address)

    beta = topology.fetch_node_via_nick_name('beta')

    assert beta

    edge = topo.Edge(alpha, beta)

    topology.add_edge(edge)

    topology.add_edge(beta, 'alpha')

    print(topology)

    topology.remove_edge(0)

    topology.remove_node(0)

    topology.clear_useless_nodes()

    print(topology)

    data = topology.serialize()
    topology.unserialize(data)


def test_analysis():
    topology = topo.Topology()

    topology.add_node('alpha', openfed.empty_address)
    topology.add_node('beta', openfed.empty_address)
    topology.add_node('gamma', openfed.empty_address)

    topology.add_edge('alpha', 'beta')
    topology.add_edge('beta', 'gamma')
    topology.add_edge('gamma', 'alpha')

    alpha_fg = topo.analysis(topology, 'alpha')
    assert len(alpha_fg) == 2

    beta_fg = topo.analysis(topology, 'beta')
    assert len(beta_fg) == 2

    gamma_fg = topo.analysis(topology, 'gamma')
    assert len(gamma_fg) == 2
