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

from openfed.core import World, follower

world = World(role=follower, dal=False, mtt=5)

print(world)

from openfed import API

openfed_api = API(
    world=world,
    state_dict=network.state_dict(keep_vars=True),
    fed_optim=fed_optim)

print(openfed_api)

from openfed.hooks import PaillierCrypto

with openfed_api:
    paillier_crypto = PaillierCrypto(public_key)
    print(paillier_crypto)

from openfed.common import default_tcp_address

address = default_tcp_address

print(address)

openfed_api.build_connection(address=address)

print(openfed_api.federated_group)

import random
import time

version = 0
for outter in range(5):
    success = True
    for inner in range(2):
        openfed_api.update_version(version)
        if not openfed_api.transfer(to=False):
           print("Download failed.")
           success = False
           break
        
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
        openfed_api.update_version(version+1)
        
        if not openfed_api.transfer(to=True):
            print("Upload failed.")
            success = False
            break
        else:
            fed_optim.clear_buffer()
        
        print(f"Outter: {outter}, Inner: {inner}, version: {version}, loss: {loss:.2f}, duration: {duration:.2f}")
    if not success:
        break
    version += 1
print("Finished.")