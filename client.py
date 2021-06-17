import torch
import torch.nn as nn
import torch.optim as optim
import random

from openfed import Frontend
from openfed.optim.elastic_aux import ElasticAux

print("Connect to Server...")
frontend = Frontend()

# 创建一个模型，
net = nn.Linear(1, 1)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
elastic_aux = ElasticAux(net.parameters())

# 将模型参数告知后端
frontend.set_state_dict(net.state_dict(keep_vars=True))

for i in range(1, 6):
    print(f"Train @{i}")
    # 下载一份数据
    frontend.download()

    # 进行训练
    net(torch.randn(128, 1, 1)).sum().backward()
    elastic_aux.step()
    optimizer.step()

    # 提交数据
    frontend.pack_state(optimizer, keys=['momentum_buffer'])
    frontend.pack_state(elastic_aux)

    frontend.set_task_info({"train_instances": random.randint(1, 200)})

    frontend.upload()

frontend.finish()

print("PASS")
