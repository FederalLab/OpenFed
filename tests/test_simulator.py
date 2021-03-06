# @Author            : FederalLab
# @Date              : 2021-09-25 16:57:22
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:57:22
# Copyright (c) FederalLab. All rights reserved.
import random

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import openfed
from openfed.data import IIDPartitioner, PartitionerDataset


def main_function(props):
    props = openfed.federated.FederatedProperties.load(props)
    assert len(props) == 1
    props = props[0]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    network = nn.Linear(784, 10).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    sgd = torch.optim.SGD(
        network.parameters(), lr=1.0 if props.aggregator else 0.1)
    fed_sgd = openfed.optim.FederatedOptimizer(sgd, props.role)

    maintainer = openfed.core.Maintainer(props,
                                         network.state_dict(keep_vars=True))

    with maintainer:
        openfed.functional.device_alignment()
        if props.aggregator:
            openfed.functional.count_step(props.address.world_size - 1)

    rounds = 1
    if maintainer.aggregator:
        api = openfed.API(maintainer, fed_sgd, rounds,
                          openfed.functional.average_aggregation)
        api.run()
    else:
        mnist = MNIST(r'/tmp/', True, ToTensor(), download=True)
        fed_mnist = PartitionerDataset(
            mnist, total_parts=100, partitioner=IIDPartitioner())

        dataloader = DataLoader(
            fed_mnist,
            batch_size=10,
            shuffle=True,
            num_workers=0,
            drop_last=False)

        for _ in range(rounds):
            maintainer.step(upload=False)

            part_id = random.randint(0, 9)
            fed_mnist.set_part_id(part_id)

            network.train()
            losses = []
            for data in dataloader:
                x, y = data
                x, y = x.to(device), y.to(device)
                output = network(x.view(-1, 784))
                loss = loss_fn(output, y)

                fed_sgd.zero_grad()
                loss.backward()
                fed_sgd.step()
                losses.append(loss.item())
            loss = sum(losses) / len(losses)

            fed_sgd.round()

            maintainer.update_version()
            maintainer.package(fed_sgd)
            maintainer.step(download=False)
            fed_sgd.clear_state_dict()


@pytest.mark.run(order=0)
def test_build_centralized_topology():
    from openfed.tools.simulator import build_centralized_topology
    build_centralized_topology(3, tcp=True)


@pytest.mark.run(order=3)
def test_simulator_aggregator():
    test_build_centralized_topology()
    main_function('/tmp/aggregator.json')


@pytest.mark.run(order=3)
def test_simulator_collaborator_alpha():
    test_build_centralized_topology()
    main_function('/tmp/collaborator-1.json')


@pytest.mark.run(order=3)
def test_simulator_collaborator_beta():
    test_build_centralized_topology()
    main_function('/tmp/collaborator-2.json')
