# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from datetime import timedelta
from enum import Enum, unique

# communication timeout
DEFAULT_PG_TIMEOUT = timedelta(minutes=30)
# null progress group
NULL_PG = None

@unique
class ROLE(Enum):
    LEADER   = 'openfed_leader'
    FOLLOWER = 'openfed_follower'

leader   = ROLE.LEADER.value
follower = ROLE.FOLLOWER.value

def is_leader(role):
    return role == leader

def is_follower(role):
    return role == follower

@unique
class STATUS(Enum):
    PUSH    = "PUSH"  # push data to the other end.
    PULL    = "PULL"  # pull data from the other end.
    ZOMBIE  = "ZOMBIE"  # when there is no request.
    OFFLINE = "OFFLINE"  # offline.


push    = STATUS.PUSH.value
pull    = STATUS.PULL.value
zombie  = STATUS.ZOMBIE.value
offline = STATUS.OFFLINE.value

@unique
class CONST(Enum):
    """The following keys will used as the key of store.
    """
    OPENFED_IDENTIFY  = "OPENFED_IDENTIFY"
    OPENFED_STATUS    = "OPENFED_STATUS"
    OPENFED_TASK_INFO = 'OPENFED_TASK_INFO'
    NICK_NAME         = 'NICK_NAME'
    LEADER_RANK       = 0
    FOLLOWER_RANK     = 1

openfed_identity  = CONST.OPENFED_IDENTIFY.value
openfed_status    = CONST.OPENFED_STATUS.value
openfed_task_info = CONST.OPENFED_TASK_INFO.value
nick_name         = CONST.NICK_NAME.value
leader_rank       = CONST.LEADER_RANK.value
follower_rank     = CONST.FOLLOWER_RANK.value
