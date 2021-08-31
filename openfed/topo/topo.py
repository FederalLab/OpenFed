import warnings

from typing import List, Optional, Tuple, overload

from openfed.common import Address
from openfed.core import (FederatedGroupProperties, follower, follower_rank,
                          is_follower, is_leader, leader, leader_rank)
import torch


class Node(object):
    def __init__(self, nick_name: str, address: Address, mtt: int):
        self.nick_name = nick_name
        self.address = address
        self.mtt = mtt

    def __eq__(self, other):
        return self.nick_name == other.nick_name and self.address == other.address


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

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end


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
            if len(self.others) > 1:
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
    def federated_group_properties(self) -> FederatedGroupProperties:
        """The address in FederatedGroupProperties needs be rectified in `Topology`.
        """
        role = self.role
        nick_name = self.node.nick_name

        if self.leader:
            address = self.node.address
            mtt = self.node.mtt
        else:
            address = self.others[0].address
            mtt = self.others[0].mtt

        return FederatedGroupProperties(role, nick_name, address, mtt)


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
    def add_node(self, nick_name, address, mtt):
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

    def remove_node(self, index: int):
        assert 0 <= index < len(self.nodes)
        del self.nodes[index]

    def build_federated_group(self, node) -> Tuple[List, List]:
        # leader group
        leader_group = []
        follower_group = []
        for edge in self.edges:
            if edge.end == node:
                # leader group
                fuse = False
                for lg in leader_group:
                    if lg.add_to_group(edge):
                        fuse = True
                        break
                if not fuse:
                    lg = FederatedGroup(leader, node)
                    assert lg.add_to_group(edge)
                    leader_group.append(lg)
            elif edge.start == node:
                # follower group
                fuse = False
                for fg in follower_group:
                    if fg.add_to_group(edge):
                        fuse = True
                        break
                if not fuse:
                    fg = FederatedGroup(follower, node)
                    assert fg.add_to_group(edge)
                    follower_group.append(fg)
        return leader_group, follower_group

    def topology_analysis(self, node: Node) -> List[FederatedGroupProperties]:

        leader_group, follower_group = self.build_federated_group(node)

        # rectify the address infomation
        leader_group_props = []
        for lg in leader_group:
            world_size = len(lg.others) + 1
            rank = leader_rank if world_size == 2 else 0

            lgp = lg.federated_group_properties

            lgp.address = Address(lgp.address.backend, lgp.address.init_method,
                                  world_size, rank)

            leader_group_props.append(lgp)

        follower_group_props = []
        for fg in follower_group:
            # build the federated group for leader
            lgs, _ = self.build_federated_group(fg.others[0])
            lg = None
            for lg in lgs:
                if fg.node in lg.others:
                    break
            if lg:
                world_size = len(lg.others) + 1
                if world_size == 2:
                    rank = follower_rank
                else:
                    nick_name = [node.nick_name for node in lg.others]
                    nick_name.sort()
                    rank = -1
                    for i, n in enumerate(nick_name):
                        if n == fg.node.nick_name:
                            rank = i + 1  # rank == 0 is leader.
                            break
                    assert rank > 0
            else:
                raise RuntimeError(f"Build leader federated group failed.")

            fgp = fg.federated_group_properties

            fgp.address = Address(
                fgp.address.backend,
                fgp.address.init_method,
                world_size,
                rank,
            )
            follower_group_props.append(fgp)

        return leader_group_props + follower_group_props

    def save(self, filename):
        torch.save([self.nodes, self.edges], filename)

    def load(self, filename):
        self.nodes, self.edges = torch.load(filename)

    def __str__(self):
        pass