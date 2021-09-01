from openfed.common import Address
from openfed.core import *

fed_props = FederatedProperties(leader, 'my_leader', Address(rank=0))
# fed_props = FederatedProperties(follower, 'my_follower', Address(rank=1))

pipe = init_federated_group(fed_props)

print(len(pipe))
print(pipe[0])
