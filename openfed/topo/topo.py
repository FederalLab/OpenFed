# Copyright (c) FederalLab. All rights reserved.
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union, overload

from openfed.common import Address
from openfed.federated import (FederatedProperties, is_aggregator,
                               is_collaborator)
from openfed.utils import openfed_class_fmt, tablist


class Node(object):
    r"""Node.

    Args:
        nick_name: An unique string to identify the node.
        address: The address of node.
    """

    def __init__(self, nick_name: str, address: Address):
        self.nick_name = nick_name
        self.address = address

    def __eq__(self, other):
        return self.nick_name == other.nick_name and\
            self.address == other.address

    def __repr__(self):
        description = 'nick name: ' + self.nick_name + '\n' + str(self.address)
        return openfed_class_fmt.format(
            class_name=self.__class__.__name__, description=description)

    def serialize(self) -> Dict[str, Any]:
        return dict(
            nick_name=self.nick_name,
            address=self.address.serialize(),
        )

    @classmethod
    def unserialize(cls, data: Dict[str, Any]):
        address = Address.unserialize(data.pop('address'))
        return Node(**data, address=address)


class Edge(object):
    r"""Edge.

    Args:
        start: namely collaborator.
        end: namely aggregator.
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
        r"""Returns `True` if given node is start or end node.
        """
        return self.is_start(node) or self.is_end(node)

    def is_start(self, node: Node) -> bool:
        return self.start == node

    def is_end(self, node: Node) -> bool:
        return self.end == node

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def serialize(self) -> Dict[str, Any]:
        return dict(
            start=self.start.serialize(),
            end=self.end.serialize(),
        )

    @classmethod
    def unserialize(cls, data: Dict[str, Any]):
        start = Node.unserialize(data.pop('start'))
        end = Node.unserialize(data.pop('end'))
        return Edge(start, end)

    def __repr__(self):
        description = f'|{self.start.nick_name} -> {self.end.nick_name}.'
        return openfed_class_fmt.format(
            class_name=self.__class__.__name__, description=description)


class FederatedGroup(object):
    r"""Federated Group gathers all surrounding nodes around the given one.

    Args:
        role: The role played. `aggregator` or `collaborator`.
        node: The given node.
    """
    role: str
    node: Node
    others: List[Node]

    def __init__(self, role: str, node: Node):
        self.role = role
        self.node = node

        self.others = []

    @property
    def aggregator(self):
        return is_aggregator(self.role)

    @property
    def collaborator(self):
        return is_collaborator(self.role)

    def add_to_group(self, edge: Edge) -> bool:
        """Add an edge to federated group."""
        if self.aggregator and edge.end == self.node:
            # belong to this group
            # add the collaborator to this group
            self.others.append(edge.start)
            return True
        if self.collaborator and edge.start == self.node:
            # belong to this group
            if len(self.others) >= 1:
                # this group is not empty,
                # as a collaborator, we only allow it belongs to
                # a single aggregator node in one federated group.
                # if it belongs to more than one aggregator, you
                # need to build federated group for each one.
                return False
            else:
                self.others.append(edge.end)
                return True
        return False

    @property
    def federated_properties(self) -> FederatedProperties:
        """The address in FederatedProperties needs be rectified in
        :func:`openfed.topo.analysis`.
        """
        role = self.role
        nick_name = self.node.nick_name

        if self.aggregator:
            address = self.node.address
        else:
            address = self.others[0].address
        return FederatedProperties(role, nick_name, address)


class Topology(object):
    r"""Topology manages massive nodes and edges.
    """

    nodes: List[Node]
    edges: List[Edge]

    def __init__(self,
                 nodes: Optional[List[Node]] = None,
                 edges: Optional[List[Edge]] = None):
        super().__init__()

        self.nodes = deepcopy(nodes or [])
        self.edges = deepcopy(edges or [])

    @overload
    def add_node(self, node: Node):
        r"""Add a node to topology.

        Args:
            node: The node to be added.
        """

    @overload
    def add_node(self, nick_name, address):
        r"""Build a node and add it to topology.

        Args:
            nick_name: The name of node.
            address: The address of node.
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

    def add_node_list(self, node_list: List[Node]):
        r"""Add a list of node to topology.

        Args:
            node_list: The list of node.
        """
        for node in node_list:
            self.add_node(node)

    @overload
    def add_edge(self, edge: Edge):
        r"""Add an edge to topology.

        .. note::

            If start and end node is not contained in the topology, we will add
            them automatically.

        Args:
            edge: The edge to be added.

        """

    @overload
    def add_edge(self, start: Union[Node, str], end: Union[Node, str]):
        """Build and then add an edge to topology.

        .. note::

            If start and end node is not contained in the topology, we will add
            them automatically.

        Args:
            start: The node or nick name of start.
            end: The node or nick name of end.
        """

    def add_edge(self, *args):
        if len(args) == 1:
            edge = args[0]
        else:
            start = self.fetch_node_via_nick_name(args[0]) if isinstance(
                args[0], str) else args[0]
            end = self.fetch_node_via_nick_name(args[1]) if isinstance(
                args[1], str) else args[1]
            edge = Edge(start, end)  # type: ignore

        if edge.start not in self.nodes:
            self.add_node(edge.start)
        if edge.end not in self.nodes:
            self.add_node(edge.end)

        if edge in self.edges:
            warnings.warn(f'{edge} already exists.')
        else:
            self.edges.append(edge)

    def remove_edge(self, index: int):
        assert 0 <= index < len(self.edges)
        del self.edges[index]

    def clear_useless_nodes(self):
        r"""Removes useless nodes.

        .. note::

            This function will remove all nodes which not be contained in any
            edges.

        """
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
        r"""Removes a specified node as well as all the related edges.
        """
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

    def serialize(self) -> Dict[str, Any]:
        return dict(
            nodes=[node.serialize() for node in self.nodes],
            edges=[edge.serialize() for edge in self.edges],
        )

    @classmethod
    def unserialize(cls, data: Dict[str, Any]):
        nodes = [Node.unserialize(d) for d in data['nodes']]
        edges = [Edge.unserialize(d) for d in data['edges']]
        return Topology(nodes, edges)

    def is_edge(self, start, end):
        if start == end:
            return False
        edge = Edge(start, end)
        return edge in self.edges

    def fetch_node_via_nick_name(self, nick_name: str) -> Union[Node, None]:
        """Fetches node via its nick name.

        Args:
            nick_name: The nick name of node.

        Returns:
            Node if exist, else None.
        """
        for node in self.nodes:
            if node.nick_name == nick_name:
                return node
        else:
            return None

    def __repr__(self) -> str:
        head = [node.nick_name for node in self.nodes]
        head = [r'CO\AG'] + head
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
