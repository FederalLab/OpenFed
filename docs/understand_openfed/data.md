# Data

## FederatedDataset

In order to load the simulated federated data in a uniform way, we provide :class:`FederatedDataset`. Compared with :class:`Dataset`, it has two extra attributes:

- `part_id`: Part id to load.
- `total_parts`: The total number of parts.

## PartitionerDataset

:class:`PartitionerDataset` will divide a custom dataset according to the partitioner you selected. It is the most convenient method to generate a simulated federated dataset for testing.

For example, we can use the following piece of code to generate the Federated-MNIST:

```shell
>>> from openfed.data import IIDPartitioner, PartitionerDataset
>>> from torchvision.datasets import MNIST
>>> from torchvision.transforms import ToTensor
>>> dataset = PartitionerDataset(
    MNIST(r'/tmp/', True, ToTensor(), download=True), total_parts=10, partitioner=IIDPartitioner())
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /tmp/MNIST/raw/train-images-idx3-ubyte.gz
9913344it [00:19, 502512.54it/s]
Extracting /tmp/MNIST/raw/train-images-idx3-ubyte.gz to /tmp/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to /tmp/MNIST/raw/train-labels-idx1-ubyte.gz
29696it [00:00, 853940.49it/s]
Extracting /tmp/MNIST/raw/train-labels-idx1-ubyte.gz to /tmp/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to /tmp/MNIST/raw/t10k-images-idx3-ubyte.gz
1649664it [00:04, 406894.94it/s]
Extracting /tmp/MNIST/raw/t10k-images-idx3-ubyte.gz to /tmp/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to /tmp/MNIST/raw/t10k-labels-idx1-ubyte.gz
5120it [00:00, 14221746.01it/s]
Extracting /tmp/MNIST/raw/t10k-labels-idx1-ubyte.gz to /tmp/MNIST/raw

Processing...
/Users/densechen/miniconda3/envs/openfed/lib/python3.7/site-packages/torchvision/datasets/mnist.py:502: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /Users/distiller/project/conda/conda-bld/pytorch_1616554799287/work/torch/csrc/utils/tensor_numpy.cpp:143.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
Done!
>>> from openfed.data.utils import samples_distribution
>>> samples_distribution(dataset, True)
+-------+---------+---------+------+
| Parts | Samples |   Mean  | Var  |
+-------+---------+---------+------+
|   10  |  59960  | 5996.00 | 0.00 |
+-------+---------+---------+------+
[5996, 5996, 5996, 5996, 5996, 5996, 5996, 5996, 5996, 5996]
```

## Partitioner

:class:`Partitioner` can generate a non-iid distribution datasets easily. We provide three different ways, i.e., `PowerLawPartitioner`, `DirichletPartitioner`, `IIDPartitioner`.
