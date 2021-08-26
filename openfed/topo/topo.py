from collections import defaultdict
from typing import Any, Dict

from openfed.utils import tablist


class FederatedGroup(object):
    def __init__(self, role, node):
        self.role = role
        self.node = node

        self.others = []
        self.metas = []

    def merge(self, other, meta):
        if meta in self.metas:
            self.others.append(other)
            self.metas.append(meta)
            return True
        else:
            return False

class Topology(object):
    """
    start -> end: follower -> leader
    """
    graph: Dict[Any, Dict[Any, Any]]

    def __init__(self):
        super().__init__()
        self.graph = defaultdict(dict)
        self.node_list = set()
        self.node_in_degree = defaultdict(int)
        self.node_out_degree = defaultdict(int)

    def is_edge(self, start, end):
        return self.is_dir_edge(start, end) and self.is_dir_edge(end, start)

    def is_dir_edge(self, start, end):
        return end in self.graph[start]

    def is_empty_node(self, node):
        return self.node_in_degree[node] == 0 and self.node_out_degree[
            node] == 0

    def add_dir_edge(self, start, end, meta):
        if self.is_edge(start, end):
            raise RuntimeError(
                f"Edge <{start}> to <{end}> is already registered.")
        self.node_list.add(start)
        self.node_list.add(end)

        self.graph[start][end] = meta

        self.node_out_degree[start] += 1
        self.node_in_degree[end] += 1

    def add_edge(self, start, end, meta):
        if self.is_edge(start, end):
            raise RuntimeError(
                f"Edge <{start}> to <{end}> is already registered.")
        if self.is_edge(end, start):
            raise RuntimeError(
                f"Edge <{end}> to <{start}> is already registered.")

        self.node_list.add(start)
        self.node_list.add(end)

        self.graph[start][end] = meta
        self.graph[end][start] = meta

        self.node_in_degree[start] += 1
        self.node_out_degree[start] += 1

        self.node_in_degree[end] += 1
        self.node_out_degree[end] += 1

    def remove_dir_edge(self, start, end):
        if self.is_dir_edge(start, end):
            # remove
            del self.graph[start][end]
            self.node_out_degree[start] -= 1
            self.node_in_degree[end] -= 1
            # clear empty node
            if self.is_empty_node(start):
                self.node_list.remove(start)
            if self.is_empty_node(end):
                self.node_list.remove(end)

    def remove_edge(self, start, end):
        self.remove_dir_edge(start, end)
        self.remove_dir_edge(end, start)

    def get_dir_edge(self, start, end):
        return self.graph[start][end]

    def __str__(self) -> str:
        nodes = sorted(list(self.node_list))
        head = [node for node in nodes]
        head = ['node'] + head
        data = []
        for start in nodes:
            items = [
                str(start),
            ]
            for end in nodes:
                if self.is_edge(start, end):
                    items.append('=')
                elif self.is_dir_edge(start, end):
                    items.append('-')
                else:
                    items.append('.')
            data += items
        return tablist(head=head, data=data, force_in_one_row=True)

    def federated_group(self, node):
        # follower search
        # follower is only belong to a specified leader.
        follower = []
        for other in self.node_list:
            if self.is_dir_edge(node, other):
                federated_group = FederatedGroup(follower, node)
                follower.append(federated_group)

        # leader search
        leader = []
        for other in self.node_list:
            if self.is_dir_edge(other, node):
                meta = self.get_dir_edge(other, node)
                fuse = False
                for federated_group in leader:
                    if federated_group.merge(other, meta):
                        fuse = True
                        break
                if not fuse:
                    federated_group = FederatedGroup(leader, node)
                    federated_group.merge(other, meta)

        # leader can be shared among many node.
        # we need to merge these node with the same leader as a single federated group.
        return follower + leader
