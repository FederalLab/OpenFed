from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from openfed.data import PartitionerDataset, samples_distribution


def test_power_law_partitioner():
    from openfed.data import PowerLawPartitioner

    try:
        # PyTorch 1.8 will raise an ERROR while downloading dataset.
        mnist = MNIST(r'/tmp/', True, ToTensor(), download=True)
    except ModuleNotFoundError:
        return

    dataset = PartitionerDataset(
        mnist, total_parts=100, partitioner=PowerLawPartitioner())

    samples_distribution(dataset)

    for _ in dataset:
        break


def test_dirichlet_partitioner():
    from openfed.data import DirichletPartitioner

    try:
        # PyTorch 1.8 will raise an ERROR while downloading dataset.
        mnist = MNIST(r'/tmp/', True, ToTensor(), download=True)
    except ModuleNotFoundError:
        return

    dataset = PartitionerDataset(
        mnist, total_parts=100, partitioner=DirichletPartitioner())

    samples_distribution(dataset)

    for _ in dataset:
        break


def test_iid_partitioner():
    from openfed.data import IIDPartitioner

    try:
        # PyTorch 1.8 will raise an ERROR while downloading dataset.
        mnist = MNIST(r'/tmp/', True, ToTensor(), download=True)
    except ModuleNotFoundError:
        return

    dataset = PartitionerDataset(
        mnist, total_parts=100, partitioner=IIDPartitioner())

    samples_distribution(dataset)

    for _ in dataset:
        break
