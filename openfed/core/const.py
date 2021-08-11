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

leader   = 'openfed_leader'
follower = 'openfed_follower'

def is_leader(role):
    return role == leader

def is_follower(role):
    return role == follower

push    = 'PUSH'
pull    = 'PULL'
zombie  = 'ZOMBIE'
offline = 'OFFLINE'

openfed_identity  = 'OPENFED_IDENTIFY'
openfed_status    = 'OPENFED_STATUS'
openfed_task_info = 'OPENFED_TASK_INFO'
nick_name     = 'NICK_NAME'
leader_rank   = 0
follower_rank = 1
