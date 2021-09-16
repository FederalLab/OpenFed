# Copyright (c) FederalLab. All rights reserved.
aggregator = 'openfed_aggregator'
collaborator = 'openfed_collaborator'


def is_aggregator(role: str) -> bool:
    """Returns `True` if `role` is aggregator.
    """
    return role == aggregator


def is_collaborator(role: str) -> bool:
    """Returns `True` if `role` is collaborator.
    """
    return role == collaborator


push = 'push'
pull = 'pull'
zombie = 'zombie'
offline = 'offline'

openfed_identity = 'openfed_identity'
openfed_status = 'openfed_status'
openfed_meta = 'openfed_meta'
nick_name = 'nick_name'

aggregator_rank = 0
collaborator_rank = 1
