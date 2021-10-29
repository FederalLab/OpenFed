<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

<div align=center> <img src="https://github.com/FederalLab/OpenFed/raw/main/docs/_static/image/openfed-logo.png" width="300" /> </div>

# OpenFed: A Comprehensive and Versatile Open-Source Federated Learning Framework

![GitHub last commit](https://img.shields.io/github/last-commit/FederalLab/OpenFed) [![Documentation Status](https://readthedocs.org/projects/openfed/badge/?version=latest)](https://openfed.readthedocs.io) [![PyPI](https://img.shields.io/pypi/v/OpenFed)](https://pypi.org/project/OpenFed) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/OpenFed)](https://pypi.org/project/OpenFed) [![badge](https://github.com/FederalLab/OpenFed/workflows/build/badge.svg)](https://github.com/FederalLab/OpenFed/actions) [![codecov](https://codecov.io/gh/FederalLab/OpenFed/branch/main/graph/badge.svg)](https://codecov.io/gh/FederalLab/OpenFed) [![license](https://img.shields.io/github/license/FederalLab/OpenFed.svg)](https://github.com/FederalLab/OpenFed/blob/master/LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2109.07852-red.svg)](https://arxiv.org/abs/2109.07852)

## Introduction

OpenFed is a foundational library for federated learning research and supports many research projects. It reduces the barrier to entry for both researchers and downstream users of Federated Learning by the targeted removal of existing pain points. For researchers, OpenFed provides a framework wherein new methods can be easily implemented and fairly evaluated against an extensive suite of benchmarks. For downstream users, OpenFed allows Federated Learning to be plug and play within different subject-matter contexts, removing the need for deep expertise in Federated Learning.

## Install

PyTorch >= 1.5.1, python>=3.6

**Build latest version from source**:

```shell
git clone https://github.com/FederalLab/OpenFed.git
cd OpenFed
pip install -e .
```

**Stable version**: `pip install openfed`

## Start Federated Learning in an Unprecedented Simple Way

```python
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# >>> Import OpenFed
import openfed
# <<<

# >>> Define arguments
parser = argparse.ArgumentParser(description='Simulator')
parser.add_argument('--props', type=str, default='/tmp/aggregator.json')
args = parser.parse_args()
# <<<

# >>> Load Federated Group Properties
props = openfed.federated.FederatedProperties.load(args.props)[0]
# <<<

network = nn.Linear(784, 10)
loss_fn = nn.CrossEntropyLoss()

sgd = torch.optim.SGD(
    network.parameters(), lr=1.0 if props.aggregator else 0.1)

# >>> Convert torch optimizer to federated optimizer
fed_sgd = openfed.optim.FederatedOptimizer(sgd, props.role)
# <<<

# >>> Define maintainer to maintain communication among each nodes
maintainer = openfed.core.Maintainer(props, network.state_dict(keep_vars=True))
# <<<

# >>> Auto register the hook function to maintainer
with maintainer:
    openfed.functional.device_alignment()
    if props.aggregator:
        openfed.functional.count_step(props.address.world_size - 1)
# <<<

# total rounds to simulation
rounds = 10
if maintainer.aggregator:
    # >>> API Loop as aggregator
    api = openfed.API(maintainer, fed_sgd, rounds,
                      openfed.functional.average_aggregation)
    api.run()
    # <<<
else:
    mnist = MNIST(r'/tmp/', True, ToTensor(), download=True)
    # >>> Convert to federated dataset
    fed_mnist = openfed.data.PartitionerDataset(
        mnist, total_parts=100, partitioner=openfed.data.IIDPartitioner())
    # <<<

    dataloader = DataLoader(
        fed_mnist, batch_size=10, shuffle=True, num_workers=0, drop_last=False)

    for outter in range(rounds):
        # >>> Download latest model from aggregator
        maintainer.step(upload=False)
        # <<<

        # Pick up a random federated dataset part
        part_id = random.randint(0, 9)
        fed_mnist.set_part_id(part_id)

        network.train()
        losses = []
        for data in dataloader:
            x, y = data
            output = network(x.view(-1, 784))
            loss = loss_fn(output, y)

            fed_sgd.zero_grad()
            loss.backward()
            fed_sgd.step()
            losses.append(loss.item())
        loss = sum(losses) / len(losses)

        # >>> Finish a round
        fed_sgd.round()
        # <<<

        # >>> Upload trained model and optimizer state
        maintainer.update_version()
        maintainer.package(fed_sgd)
        maintainer.step(download=False)
        # <<<

        # Clear state dict
        fed_sgd.clear_state_dict()
```

Now, save this piece of code as `run.py`, and start a simulator by:

```shell
(openfed) python -m openfed.tools.simulator --nproc 6 run.py
100%|█████████████████████████████████████████| 10/10 [00:01<00:00,  7.21it/s]
```

This command will launch 6 processes (1 for aggregator, 5 for collaborators).

## Citation

If you find this project useful in your research, please consider cite:

```bash
@misc{chen2021openfed,
      title={OpenFed: A Comprehensive and Versatile Open-Source Federated Learning Framework},
      author={Dengsheng Chen and Vince Tan and Zhilin Lu and Jie Hu},
      year={2021},
      eprint={2109.07852},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

## Contributing

We appreciate all contributions to improve OpenFed.
Please refer to [CONTRIBUTING.md](https://github.com/FederalLab/OpenFed/raw/main/CONTRIBUTING.md) for the contributing guideline.

## License

OpenFed is released under the MIT License.
