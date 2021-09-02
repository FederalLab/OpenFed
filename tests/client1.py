import os
import sys
sys.path.insert(0, '/Users/densechen/code/OpenFed')
import openfed

# build a topology
import openfed

server_node = openfed.topo.Node('server', openfed.default_tcp_address)
client_alpha = openfed.topo.Node('client-alpha', openfed.empty_address)
client_beta = openfed.topo.Node('client-beta', openfed.empty_address)

topology = openfed.topo.Topology()
topology.add_edge(client_alpha, server_node)
topology.add_edge(client_beta, server_node)

federated_group_props = topology.analysis(server_node)[0]
