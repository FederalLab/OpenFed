import os
import sys
sys.path.insert(0, '/Users/densechen/code/OpenFed/')


import torch
import os

if not os.path.isfile('/tmp/public.key'):
    raise FileNotFoundError("Public Key is not found. Generate it using the `paillier_crypto_server` script.")
else:
    public_key = torch.load('/tmp/public.key')
    print("Load public key from /tmp/public.key")
    print(public_key)


from openfed.data import IIDPartitioner, PartitionerDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

dataset = PartitionerDataset(
    MNIST(r'/tmp/', True, ToTensor(), download=True), total_parts=10, partitioner=IIDPartitioner())

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=False)

import torch.nn as nn

network = nn.Linear(784, 10)
loss_fn = nn.CrossEntropyLoss()

import torch

from openfed.optim import build_fed_optim

optim = torch.optim.SGD(network.parameters(), lr=0.1)
fed_optim = build_fed_optim(optim)

print(fed_optim)


import openfed

server_node = openfed.topo.Node('server', openfed.default_tcp_address, mtt=5)
client = openfed.topo.Node('client', openfed.empty_address, mtt=5)

topology = openfed.topo.Topology()
topology.add_edge(client, server_node)

federated_group_props = topology.topology_analysis(client)[0]

print(federated_group_props)


from openfed import API

openfed_api = API(
    state_dict=network.state_dict(keep_vars=True),
    fed_optim=fed_optim)

print(openfed_api)

from openfed.hooks import PaillierCrypto

with openfed_api:
    paillier_crypto = PaillierCrypto(public_key)
    print(paillier_crypto)

openfed_api.build_connection(federated_group_props)

import random
import time

version = 0
for outter in range(5):
    for inner in range(2):
        openfed_api.update_version(version)
        openfed_api.step(upload=False)
        
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
        loss = sum(losses)/len(losses)
        duration = toc-tic

        fed_optim.round()

        openfed_api.update_version(version + 1)
        openfed_api.step(download=False)
        fed_optim.clear_buffer()
        
        print(f"Outter: {outter}, Inner: {inner}, version: {version}, loss: {loss:.2f}, duration: {duration:.2f}")
    version += 1
print("Finished.")