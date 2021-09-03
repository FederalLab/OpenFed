# Copyright (c) FederalLab. All rights reserved.
import cmd

from openfed.common import Address, empty_address
from openfed.topo import Edge, Node, Topology
from openfed.utils import openfed_title


class TopoBuilder(cmd.Cmd):
    r"""

    Example::

        >>> python -m openfed.tools.topo_builder
        A script to build topology.
        <OpenFed>: help

        Documented commands (type help <topic>):
        ========================================
        add_node    clear_useless_nodes  help  plot         remove_node
        build_edge  exit                 load  remove_edge  save       

        <OpenFed>: add_node
        Nick Name
        server
        Does this node requires address? (Y/n)
        y
        Backend (gloo, mpi, nccl)
        gloo
        Init method i.e., 'tcp://localhost:1994', file:///tmp/openfed.sharedfile)
        tcp://localhost:1995
        <OpenFed> Node
        nick name: server
        <OpenFed> Address
        +---------+---------------------+------------+------+
        | backend |     init_method     | world_size | rank |
        +---------+---------------------+------------+------+
        |   gloo  | tcp://localhost:... |     2      |  -1  |
        +---------+---------------------+------------+------+


        <OpenFed>: add_node
        Nick Name
        alpha
        Does this node requires address? (Y/n)
        n
        <OpenFed> Node
        nick name: alpha
        <OpenFed> Address
        +---------+-------------+------------+------+
        | backend | init_method | world_size | rank |
        +---------+-------------+------------+------+
        |   null  |     null    |     2      |  -1  |
        +---------+-------------+------------+------+


        <OpenFed>: add_node
        Nick Name
        beta
        Does this node requires address? (Y/n)
        n
        <OpenFed> Node
        nick name: beta
        <OpenFed> Address
        +---------+-------------+------------+------+
        | backend | init_method | world_size | rank |
        +---------+-------------+------------+------+
        |   null  |     null    |     2      |  -1  |
        +---------+-------------+------------+------+


        <OpenFed>: plot
        +--------+--------+-------+------+
        | fl\lr  | server | alpha | beta |
        +--------+--------+-------+------+
        | server |   .    |   .   |  .   |
        | alpha  |   .    |   .   |  .   |
        |  beta  |   .    |   .   |  .   |
        +--------+--------+-------+------+
        <OpenFed>: add_edge
        *** Unknown syntax: add_edge
        <OpenFed>: build_edge
        Start node's nick name
        alpha
        End node's nick name
        server
        <OpenFed> Edge
        |alpha -> server.

        <OpenFed>: build_edge
        Start node's nick name
        beta
        End node's nick name
        server
        <OpenFed> Edge
        |beta -> server.

        <OpenFed>: plot
        +--------+--------+-------+------+
        | fl\lr  | server | alpha | beta |
        +--------+--------+-------+------+
        | server |   .    |   .   |  .   |
        | alpha  |   ^    |   .   |  .   |
        |  beta  |   ^    |   .   |  .   |
        +--------+--------+-------+------+
        <OpenFed>: add_node
        Nick Name
        gamma
        Does this node requires address? (Y/n)
        Y
        Backend (gloo, mpi, nccl)
        mpi
        Init method i.e., 'tcp://localhost:1994', file:///tmp/openfed.sharedfile)
        file:///tmp/gamma
        <OpenFed> Node
        nick name: gamma
        <OpenFed> Address
        +---------+-------------------+------------+------+
        | backend |    init_method    | world_size | rank |
        +---------+-------------------+------------+------+
        |   mpi   | file:///tmp/gamma |     2      |  -1  |
        +---------+-------------------+------------+------+


        <OpenFed>: build_edge
        Start node's nick name
        alpha
        End node's nick name
        beta
        <OpenFed> Edge
        |alpha -> beta.

        <OpenFed>: build_edge
        Start node's nick name
        gamma
        End node's nick name
        beta
        <OpenFed> Edge
        |gamma -> beta.

        <OpenFed>: plot
        +--------+--------+-------+------+-------+
        | fl\lr  | server | alpha | beta | gamma |
        +--------+--------+-------+------+-------+
        | server |   .    |   .   |  .   |   .   |
        | alpha  |   ^    |   .   |  ^   |   .   |
        |  beta  |   ^    |   .   |  .   |   .   |
        | gamma  |   .    |   .   |  ^   |   .   |
        +--------+--------+-------+------+-------+
        <OpenFed>: save
        Filename:
        topology
        +--------+--------+-------+------+-------+
        | fl\lr  | server | alpha | beta | gamma |
        +--------+--------+-------+------+-------+
        | server |   .    |   .   |  .   |   .   |
        | alpha  |   ^    |   .   |  ^   |   .   |
        |  beta  |   ^    |   .   |  .   |   .   |
        | gamma  |   .    |   .   |  ^   |   .   |
        +--------+--------+-------+------+-------+
        <OpenFed>: exit
    """
    intro = f"A script to build topology."

    @property
    def prompt(self):
        return f'{openfed_title}: '

    def __init__(self):
        super().__init__()
        self.topology = Topology()

    def do_add_node(self, *args, **kwargs):
        r"""Add a new node to the topology.
        """
        nick_name = input("Nick Name\n")
        while True:
            flag = input("Does this node requires address? (Y/n)\n")
            flag = flag.lower()
            if flag not in ['y', 'n']:
                print("Invalid choice.")
            else:
                break
        if flag == 'y':
            while True:
                backend = input("Backend (gloo, mpi, nccl)\n")
                if backend in ['gloo', 'mpi', 'nccl']:
                    break
                else:
                    print("Invalid backend.")
            while True:
                init_method = input(
                    "Init method i.e., 'tcp://localhost:1994', file:///tmp/openfed.sharedfile)\n"
                )
                try:
                    address = Address(backend, init_method)
                    break
                except Exception as e:
                    print(e)
                    print("Invalid address.")
        else:
            address = empty_address
        node = Node(nick_name, address)
        print(node)

        self.topology.add_node(node)

    def do_build_edge(self, *args, **kwargs):
        r"""Build an edge between two nodes.
        """
        while True:
            start = input("Start node's nick name\n")
            start_node = self.topology.fetch_node_via_nick_name(start)
            if start_node is not None:
                break
            else:
                print("Invalid start node.")
                return

        while True:
            end = input("End node's nick name\n")
            end_node = self.topology.fetch_node_via_nick_name(end)
            if end_node is not None:
                break
            else:
                print("Invalid end node.")
                return

        edge = Edge(start_node, end_node)
        print(edge)

        self.topology.add_edge(edge)

    def do_save(self, *args, **kwargs):
        r"""Save topology to disk.
        """
        filename = input("Filename:\n")
        self.topology.save(filename)

        print(self.topology)

    def do_load(self, *args, **kwargs):
        r"""Load topology from disk.
        """
        filename = input("Filename:\n")
        self.topology.load(filename)
        print(self.topology)

    def do_plot(self, *args, **kwargs):
        r"""Plot topology.
        """
        print(self.topology)

    def do_remove_edge(self, *args, **kwargs):
        r"""Removes an edge from the topology.
        """
        for i, e in enumerate(self.topology.edges):
            print(f"{i}\n")
            print(e)
        while True:
            edge_idx = input(
                "The index of the edge to remove (-1 to quite.)\n")
            edge_idx = int(edge_idx)
            if edge_idx == -1:
                return
            else:
                try:
                    self.topology.remove_edge(edge_idx)
                    return
                except Exception as e:
                    print(e)
                    print("Removed edge failed.")

    def do_remove_node(self, *args, **kwargs):
        r"""Removes a node from the topology.
        """
        for i, n in enumerate(self.topology.nodes):
            print(f"{i}\n")
            print(n)
        while True:
            node_idx = input(
                "The index of the node to remove (-1 to quite.)\n")
            node_idx = int(node_idx)
            if node_idx == -1:
                return
            else:
                try:
                    self.topology.remove_node(node_idx)
                except Exception as e:
                    print(e)
                    print("Removed edge failed.")

    def do_clear_useless_nodes(self, *args, **kwargs):
        r"""Clears all invalid nodes from the topology.
        """
        self.topology.clear_useless_nodes()

        print(self.topology)

    def do_exit(self, *args, **kwargs):
        r"""Exits this script."""
        exit(0)


if __name__ == "__main__":
    topo_builder = TopoBuilder()
    topo_builder.cmdloop()
