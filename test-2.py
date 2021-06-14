from openfed import pack
import openfed
from openfed.pack.package import Package

import torch

linear = torch.nn.Linear(1, 1)
optim = torch.optim.Adam(linear.parameters(), lr=0.1)

linear(torch.randn(1, 1)).backward()

optim.step()

package = Package(linear.state_dict(keep_vars=True))

package.pack_optimizer_state(optim, state_keys=["exp_avg_sq", "exp_avg"])

# 交付给底层
handout = package.handout()

# 从底层接收
package.handin(handout)

# 反向解析
package.unpack_optimizer_state(optim, state_keys=["exp_avg_sq", "exp_avg"])