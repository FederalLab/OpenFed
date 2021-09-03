# Copyright (c) FederalLab. All rights reserved.
import warnings
from typing import List, Union, overload

import torch
from openfed.common import Address
from openfed.federated import FederatedProperties, is_follower, is_leader
from openfed.utils import openfed_class_fmt, tablist


class Node(object):
    def __init__(self, nick_name: str, address: Address):
        self.nick_name = nick_name
        self.address = address

    def __eq__(self, other):
        return self.nick_name == other.nick_name and self.address == other.address

    def __repr__(self):
        description = "nick name: " + self.nick_name + '\n' + str(self.address)
        return openfed_class_fmt.format(class_name=self.__class__.__name__,
                                        description=description)


class Edge(object):
    """Edge, start node will be regarded as follower, end node will be regarded as leader.
    """
    def __init__(
        self,
        start: Node,
        end: Node,
    ):
        assert start != end
        self.start = start
        self.end = end

    def in_edge(self, node: Node) -> bool:
        return self.start == node or self.end == node

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __repr__(self):
        description = f"|{self.start.nick_name} -> {self.end.nick_name}."
        return openfed_class_fmt.format(class_name=self.__class__.__name__,
                                        description=description)


class FederatedGroup(object):
    role: str
    node: Node
    others: List[Node]

    def __init__(self, role: str, node: Node):
        self.role = role
        self.node = node

        self.others = []

    @property
    def leader(self):
        return is_leader(self.role)

    @property
    def follower(self):
        return is_follower(self.role)

    def add_to_group(self, edge: Edge) -> bool:
        """Add an edge to federated group.
        """
        if self.leader and edge.end == self.node:
            # belong to this group
            # add the follower to this group
            self.others.append(edge.start)
            return True
        if self.follower and edge.start == self.node:
            # belong to this group
            if len(self.others) >= 1:
                # this group is not empty,
                # as a follower, we only allow it belongs to
                # a single leader node in one federated group.
                # if it belongs to more than one leader, you
                # need to build federated group for each one.
                return False
            else:
                self.others.append(edge.end)
                return True
        return False

    @property
    def federated_properties(self) -> FederatedProperties:
        """The address in FederatedProperties needs be rectified in `Topology`.
        """
        role = self.role
        nick_name = self.node.nick_name

        if self.leader:
            address = self.node.address
        else:
            address = self.others[0].address
        return FederatedProperties(role, nick_name, address)


class Topology(object):
    nodes: List[Node]
    edges: List[Edge]

    def __init__(self):
        super().__init__()

        self.nodes = []
        self.edges = []

    @overload
    def add_node(self, node: Node):
        """Add a node to topology.
        """

    @overload
    def add_node(self, nick_name, address):
        """Build a node and add it to topology
        """

    def add_node(self, *args):
        if len(args) == 1:
            node = args[0]
            assert isinstance(node, Node)
        else:
            node = Node(*args)

        if node in self.nodes:
            warnings.warn(f'{node} already exists.')
        else:
            self.nodes.append(node)

    @overload
    def add_edge(self, edge: Edge):
        """Add a new edge to topology.
        If start and end node is not existing, we will add them first.
        """

    @overload
    def add_edge(self, start: Node, end: Node):
        """Build and then add a new edge to topology.
        If start and end node is not existing, we will add them first.
        """

    def add_edge(self, *args):
        if len(args) == 1:
            edge = args[0]
        else:
            edge = Edge(*args)

        if edge.start not in self.nodes:
            self.add_node(edge.start)
        if edge.end not in self.nodes:
            self.add_node(edge.end)

        if edge in self.edges:
            warnings.warn(f"{edge} already exists.")
        else:
            self.edges.append(edge)

    def remove_edge(self, index: int):
        assert 0 <= index < len(self.edges)
        del self.edges[index]

    def clear_useless_nodes(self):
        useless_idx = []
        for idx, node in enumerate(self.nodes):
            useless = True
            for edge in self.edges:
                if edge.in_edge(node):
                    useless = False
                    break
            if useless:
                useless_idx.append(idx)
        useless_idx.reverse()
        for idx in useless_idx:
            self.remove_node(idx)

    def remove_node(self, index: int):
        node = self.nodes[index]
        invalid_edge_idx = []
        for idx, edge in enumerate(self.edges):
            if edge.in_edge(node):
                invalid_edge_idx.append(idx)
        invalid_edge_idx.reverse()
        for idx in invalid_edge_idx:
            self.remove_edge(idx)

        assert 0 <= index < len(self.nodes)
        del self.nodes[index]

    def save(self, filename):
        torch.save([self.nodes, self.edges], filename)
        with open(filename + '.txt', 'w') as f:
            f.write(str(self))

    def load(self, filename):
        self.nodes, self.edges = torch.load(filename)
        return self

    def is_edge(self, start, end):
        if start == end:
            return False
        edge = Edge(start, end)
        return edge in self.edges

    def fetch_node_via_nick_name(self, nick_name: str) -> Union[Node, None]:
        for node in self.nodes:
            if node.nick_name == nick_name:
                return node
        else:
            return None

    def __repr__(self) -> str:
        head = [node.nick_name for node in self.nodes]
        head = [r'fl\lr'] + head
        data = []
        for start in self.nodes:
            items = [
                start.nick_name,
            ]
            for end in self.nodes:
                if self.is_edge(start, end):
                    items.append('^')
                else:
                    items.append('.')
            data += items
        return tablist(head=head, data=data, force_in_one_row=True)
