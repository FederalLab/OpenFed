from typing import List, Tuple, Union

from openfed.common import Address
from openfed.federated import (FederatedProperties, collaborator, collaborator_rank,
                               aggregator, aggregator_rank)

from .topo import FederatedGroup, Node, Topology


def build_federated_group(
        topo: Topology,
        node: Node) -> Tuple[List[FederatedGroup], List[FederatedGroup]]:
    r"""Build the federated group for node.

    Args:
        topo: The topology map contains related information.
        node: The node to build the federated group.
    """
    # aggregator group
    aggregator_group = []
    collaborator_group = []
    for edge in topo.edges:
        if edge.end == node:
            # aggregator group
            fuse = False
            for lg in aggregator_group:
                if lg.add_to_group(edge):
                    fuse = True
                    break
            if not fuse:
                lg = FederatedGroup(aggregator, node)
                assert lg.add_to_group(edge)
                aggregator_group.append(lg)
        elif edge.start == node:
            # collaborator group
            fuse = False
            for fg in collaborator_group:
                if fg.add_to_group(edge):
                    fuse = True
                    break
            if not fuse:
                fg = FederatedGroup(collaborator, node)
                assert fg.add_to_group(edge)
                collaborator_group.append(fg)
    return aggregator_group, collaborator_group


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

    aggregator_group, collaborator_group = build_federated_group(topo, node)

    # rectify the address infomation
    aggregator_group_props = []
    for lg in aggregator_group:
        world_size = len(lg.others) + 1
        rank = aggregator_rank if world_size == 2 else 0

        lgp = lg.federated_properties

        lgp.address = Address(lgp.address.backend, lgp.address.init_method,
                              world_size, rank)

        aggregator_group_props.append(lgp)

    collaborator_group_props = []
    for fg in collaborator_group:
        # build the federated group for aggregator
        lgs, _ = build_federated_group(topo, fg.others[0])
        lg = None
        for lg in lgs:
            if fg.node in lg.others:
                break
        if lg:
            world_size = len(lg.others) + 1
            if world_size == 2:
                rank = collaborator_rank
            else:
                nick_name = [node.nick_name for node in lg.others]
                nick_name.sort()
                rank = -1
                for i, n in enumerate(nick_name):
                    if n == fg.node.nick_name:
                        rank = i + 1  # rank == 0 is aggregator.
                        break
                assert rank > 0
        else:
            raise RuntimeError(f"Build aggregator federated group failed.")

        fgp = fg.federated_properties

        fgp.address = Address(
            fgp.address.backend,
            fgp.address.init_method,
            world_size,
            rank,
        )
        collaborator_group_props.append(fgp)

    return aggregator_group_props + collaborator_group_props
