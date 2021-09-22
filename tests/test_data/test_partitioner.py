from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from openfed.data import PartitionerDataset, samples_distribution


def test_power_law_partitioner():
    from openfed.data import PowerLawPartitioner

    mnist = MNIST(r'/tmp/', True, ToTensor(), download=True)

    dataset = PartitionerDataset(
        mnist, total_parts=100, partitioner=PowerLawPartitioner())

    samples_distribution(dataset)

    for _ in dataset:
        break


def test_dirichlet_partitioner():
    from openfed.data import DirichletPartitioner

    mnist = MNIST(r'/tmp/', True, ToTensor(), download=True)

    dataset = PartitionerDataset(
        mnist, total_parts=100, partitioner=DirichletPartitioner())

    samples_distribution(dataset)

    for _ in dataset:
        break


def test_iid_partitioner():
    from openfed.data import IIDPartitioner

    mnist = MNIST(r'/tmp/', True, ToTensor(), download=True)

    dataset = PartitionerDataset(
        mnist, total_parts=100, partitioner=IIDPartitioner())

    samples_distribution(dataset)

    for _ in dataset:
        break
