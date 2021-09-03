from typing import List, Tuple, Union

from openfed.common import Address
from openfed.federated import (FederatedProperties, follower, follower_rank,
                               leader, leader_rank)

from .topo import FederatedGroup, Node, Topology


def build_federated_group(
        topo: Topology,
        node: Node) -> Tuple[List[FederatedGroup], List[FederatedGroup]]:
    r"""Build the federated group for node.

    Args:
        topo: The topology map contains related information.
        node: The node to build the federated group.
    """
    # leader group
    leader_group = []
    follower_group = []
    for edge in topo.edges:
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


def analysis(topo: Topology, node: Union[Node,
                                         str]) -> List[FederatedProperties]:
    r"""Build the federated group for node.

    Args:
        topo: The topology map contains related information.
        node: The node to build the federated group. If string is provided, we 
            will use the string as the nick name of the node.
    """
    if isinstance(node, str):
        node = topo.fetch_node_via_nick_name(node)  # type: ignore
        assert node, 'Invalid node.'
    assert isinstance(node, Node)

    leader_group, follower_group = build_federated_group(topo, node)

    # rectify the address infomation
    leader_group_props = []
    for lg in leader_group:
        world_size = len(lg.others) + 1
        rank = leader_rank if world_size == 2 else 0

        lgp = lg.federated_properties

        lgp.address = Address(lgp.address.backend, lgp.address.init_method,
                              world_size, rank)

        leader_group_props.append(lgp)

    follower_group_props = []
    for fg in follower_group:
        # build the federated group for leader
        lgs, _ = build_federated_group(topo, fg.others[0])
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

        fgp = fg.federated_properties

        fgp.address = Address(
            fgp.address.backend,
            fgp.address.init_method,
            world_size,
            rank,
        )
        follower_group_props.append(fgp)

    return leader_group_props + follower_group_props
