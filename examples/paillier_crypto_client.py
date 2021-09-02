import os
import sys

sys.path.insert(0, '/Users/densechen/code/OpenFed/')

import os

import torch
from openfed.hooks import PublicKey

if not os.path.isfile('/tmp/public.key'):
    raise FileNotFoundError(
        "Public Key is not found. Generate it using the `paillier_crypto_server` script."
    )
else:
    public_key = PublicKey.load('/tmp/public.key')
    print("Load public key from /tmp/public.key")
print(public_key)

from openfed.data import IIDPartitioner, PartitionerDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

dataset = PartitionerDataset(MNIST(r'/tmp/', True, ToTensor(), download=True),
                             total_parts=10,
                             partitioner=IIDPartitioner())

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset,
                        batch_size=10,
                        shuffle=True,
                        num_workers=0,
                        drop_last=False)

import torch.nn as nn

network = nn.Linear(784, 10)
loss_fn = nn.CrossEntropyLoss()

import torch
from openfed.core import follower
from openfed.optim import FederatedOptimizer

optim = torch.optim.SGD(network.parameters(), lr=0.1)
fed_optim = FederatedOptimizer(optim, role=follower)

print(fed_optim)

import openfed

server_node = openfed.topo.Node('server', openfed.default_tcp_address)
client = openfed.topo.Node('client', openfed.empty_address)

topology = openfed.topo.Topology()
topology.add_edge(client, server_node)

fed_props = topology.analysis(client)[0]

print(fed_props)

from openfed.maintainer import Maintainer

mt = Maintainer(fed_props, network.state_dict(keep_vars=True))

print(mt)

with mt:
    openfed.hooks.paillier(public_key)

import random
import time

version = 0
for outter in range(5):
    for inner in range(2):
        mt.update_version(version)
        mt.step(upload=False)

        part_id = random.randint(0, 9)
        print(f"Select part_id={part_id}")
        dataset.set_part_id(part_id)

        network.train()
        losses = []
        tic = time.time()
        for data in dataloader:
            x, y = data
            output = network(x.view(-1, 784))
            loss = loss_fn(output, y)

            fed_optim.zero_grad()
            loss.backward()
            fed_optim.step()
            losses.append(loss.item())
        toc = time.time()
        loss = sum(losses) / len(losses)
        duration = toc - tic

        fed_optim.round()

        mt.update_version(version + 1)
        mt.package(fed_optim)
        mt.step(download=False)
        fed_optim.clear_state_dict()

        print(
            f"Outter: {outter}, Inner: {inner}, version: {version}, loss: {loss:.2f}, duration: {duration:.2f}"
        )
    version += 1
print("Finished.")
