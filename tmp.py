from openfed.common import build_address
from openfed.topo import Node, Topology

if __name__ == '__main__':
    topology = Topology()

    node_a = Node('a', build_address('gloo', 'tcp://127.0.0.1:1995'), 5)
    node_b = Node('b', build_address('gloo', 'tcp://127.0.0.1:1996'), 5)
    node_c = Node('c', build_address('gloo', 'tcp://127.0.0.1:1997'), 5)

    topology.add_edge(node_a, node_b)
    topology.add_edge(node_c, node_b)
    topology.add_edge(node_c, node_a)

    federated_group_props = topology.topology_analysis(node_c)
    for fgp in federated_group_props:
        print(fgp)

    print(topology)
