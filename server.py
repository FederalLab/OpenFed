import torch.nn as nn
import torch.optim as optim
from openfed import Backend
from openfed.aggregate import ElasticAggregator
import openfed

openfed.DEBUG.set_debug()
print("Connect to Client...")
backend = Backend(address=openfed.default_address_lists[:3])

# 创建一个模型，
net = nn.Linear(1, 1)

# 将模型参数告知后端
backend.set_state_dict(net.state_dict(keep_vars=True))

# 选择一个聚合方式，并且告知：我们除了收集必须的数据以外，还会收集momentum_buffer的数据.
aggregator = ElasticAggregator(net.parameters(), other_keys="momentum_buffer")
backend.set_aggregator(aggregator)

# 选择一个服务器端的优化器
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
backend.set_optimizer(optimizer)

# 开始进入循环
backend.run()

backend.finish()

print("PASS")
