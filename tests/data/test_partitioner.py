from openfed.data.partitioner import (DirichletPartitioner, IIDPartitioner,
                                      Partitioner, PowerLawPartitioner)


def test_partitioner():
    Partitioner()
    PowerLawPartitioner()
    DirichletPartitioner()
    IIDPartitioner()
