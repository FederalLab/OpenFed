# Copyright (c) FederalLab. All rights reserved.
leader = 'openfed_leader'
follower = 'openfed_follower'


def is_leader(role: str) -> bool:
    """Returns `True` if `role` is leader.
    """
    return role == leader


def is_follower(role: str) -> bool:
    """Returns `True` if `role` is follower.
    """
    return role == follower


push = 'push'
pull = 'pull'
zombie = 'zombie'
offline = 'offline'

openfed_identity = 'openfed_identity'
openfed_status = 'openfed_status'
openfed_meta = 'openfed_meta'
nick_name = 'nick_name'

leader_rank = 0
follower_rank = 1
