from openfed.topo import TopoGraph

if __name__ == '__main__':
    topo_graph = TopoGraph()

    topo_graph.add_edge('a', 'b', '*')
    topo_graph.add_edge('d', 'f', '*')
    topo_graph.add_dir_edge('a', 'f', '+')
    topo_graph.add_dir_edge('c', 'h', '+')

    print(topo_graph)