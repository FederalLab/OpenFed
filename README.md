<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

<div align=center>
<img src="https://github.com/FederalLab/OpenFed/raw/main/docs/_static/image/openfed-logo.png" width="300" />
</div>

![GitHub last commit](https://img.shields.io/github/last-commit/FederalLab/OpenFed) [![Documentation Status](https://readthedocs.org/projects/openfed/badge/?version=latest)](https://openfed.readthedocs.io) [![PyPI](https://img.shields.io/pypi/v/OpenFed)](https://pypi.org/project/OpenFed) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/OpenFed)](https://pypi.org/project/OpenFed) [![badge](https://github.com/FederalLab/OpenFed/workflows/build/badge.svg)](https://github.com/FederalLab/OpenFed/actions) [![codecov](https://codecov.io/gh/FederalLab/OpenFed/branch/main/graph/badge.svg)](https://codecov.io/gh/FederalLab/OpenFed) [![license](https://img.shields.io/github/license/FederalLab/OpenFed.svg)](https://github.com/FederalLab/OpenFed/blob/master/LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2109.07852-red.svg)](https://arxiv.org/abs/2109.07852)

**NOTE**: Current version is unstable, and we will release the first stable version very soon.

## Introduction

OpenFed is a foundational library for federated learning research and supports many research projects as below:

- [benchmark-lightly](https://github.com/FederalLab/benchmark-lightly): FederalLab's simulation benchmark.
- [openfed-cv](https://github.com/FederalLab/openfed-cv): FederalLab's toolkit and benchmark for computer vision in federated learning. This toolkit is based on [mmcv](https://github.com/open-mmlab/mmcv/), and provides the federated learning for following tasks:
  - [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
  - [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
  - [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
  - [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
  - [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
  - [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
  - [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
  - [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
  - [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition and understanding toolbox.
  - [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [openfed-finance](https://github.com/FederalLab/openfed-finance): FederalLab's toolbox and benchmark for finance data analysis in federated learning.
- [openfed-medical](https://github.com/FederalLab/openfed-medical): FederalLab's toolbox and benchmark for medical data analysis in federated learning. It is based on [MONAI](https://github.com/Project-MONAI/MONAI).
- [openfed-nlp](https://github.com/FederalLab/openfed-nlp): FederalLab's toolbox and benchmark for natural language processing in federated learning. It is based on [transformers](https://github.com/huggingface/transformers).
- [openfed-rl](https://github.com/FederalLab/openfed-rl): FederalLab's toolbox and benchmark for reinforcement learning in federated learning. It is based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

In addition, we also provide a toolkit for better compatibility with following libraries, so that you can use OpenFed with those libraries without obstacles and more easily:

- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning): The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.
- [mmcv](https://github.com/open-mmlab/mmcv): MMCV is a foundational library for computer vision research and supports many research projects.

## Install

PyTorch >= 1.5.1, python>=3.6

**Stable version**: `pip install openfed`

**Latest version**: `pip install -e git+https://github.com/FederalLab/OpenFed.git`

## Start Federated Learning In An Unprecedented Simple Way

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

Now, save the piece of code as `run.py`, and you can use the provided script to start a simulator by:

```shell
(openfed) python -m openfed.tools.simulator --nproc 6 run.py
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  7.21it/s]
```

This command will launch 6 processes (1 for aggregator, 5 for collaborators).

## Citation

If you find this project useful in your research, please consider cite:

```bash
@misc{OpenFed,
Author = {Chen Dengsheng},
Title = {OpenFed: An Open-Source Security and Privacy Guaranteed Federated Learning Framework},
Year = {2021},
Eprint = {arXiv:2109.07852},
}
```

## Contributing

We appreciate all contributions to improve OpenFed. Please refer to [CONTRIBUTUNG.md](https://github.com/FederalLab/OpenFed/raw/main/CONTRIBUTING.md) for the contributing guideline.

## License

OpenFed is released under the MIT License.
