# Copyright (c) FederalLab. All rights reserved.
from datetime import timedelta
from threading import Lock

# null progress group
NULL_PG = None

leader = 'openfed_leader'
follower = 'openfed_follower'


def is_leader(role):
    return role == leader


def is_follower(role):
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

# Default process group wide timeout, if applicable.
# This only applies to the gloo and nccl backends
# (only if NCCL_BLOCKING_WAIT or NCCL_ASYNC_ERROR_HANDLING is set to 1).
# To make an attempt at backwards compatibility with THD, we use an
# extraordinarily high default timeout, given that THD did not have timeouts.
default_pg_timeout = timedelta(seconds=100)

openfed_lock = Lock()
