Paillier Crypto
===============

Generate Public and Private Key
-------------------------------

.. code:: ipython3

    import os
    
    import torch
    
    from openfed.functional import PrivateKey, key_gen
    
    if not os.path.isfile('/tmp/public.key') or not os.path.isfile(
            '/tmp/private.key'):
        public_key, private_key = key_gen()
        public_key.save('/tmp/public.key')
        private_key.save('/tmp/private.key')
        print("Save new key to /tmp/public.key and /tmp/private.key")
    else:
        private_key = PrivateKey.load('/tmp/private.key')
        print("Load private key from /tmp/private.key")
    print(private_key)


.. parsed-literal::

    Load private key from /tmp/private.key
    [0;34m<OpenFed>[0m [0;35mPrivateKey[0m
    +-------+------+-----------+----+-------+------------+------------+
    | n_lwe | bits | bits_safe | l  | bound |     p      |     q      |
    +-------+------+-----------+----+-------+------------+------------+
    |  3000 |  32  |     24    | 64 |   8   | 4294967297 | 4294967296 |
    +-------+------+-----------+----+-------+------------+------------+
    


Network
-------

.. code:: ipython3

    import torch.nn as nn
    
    network = nn.Linear(784, 10)
    loss_fn = nn.CrossEntropyLoss()

Optimizer
---------

.. code:: ipython3

    import torch
    from openfed.federated import aggregator
    from openfed.optim import FederatedOptimizer
    
    optim = torch.optim.SGD(network.parameters(), lr=1.0)
    fed_optim = FederatedOptimizer(optim, role=aggregator)
    print(fed_optim)


.. parsed-literal::

    [0;34m<OpenFed>[0m [0;35mFederatedOptimizer[0m
    SGD (
    Parameter Group 0
        dampening: 0
        lr: 1.0
        momentum: 0
        nesterov: False
        weight_decay: 0
    )
    


Topology
--------

.. code:: ipython3

    from openfed.topo import Node, Topology, analysis
    import openfed
    
    server_node = Node('server', openfed.default_tcp_address)
    client = Node('client', openfed.empty_address)
    
    topology = Topology()
    topology.add_edge(client, server_node)
    
    fed_props = analysis(topology, server_node)[0]
    
    print(fed_props)


.. parsed-literal::

    [0;34m<OpenFed>[0m [0;35mFederatedProperties[0m
    +----------------+-----------+
    |      role      | nick_name |
    +----------------+-----------+
    | openfed_aggregator |   server  |
    +----------------+-----------+
    [0;34m<OpenFed>[0m [0;35mAddress[0m
    +---------+---------------------+------------+------+
    | backend |     init_method     | world_size | rank |
    +---------+---------------------+------------+------+
    |   gloo  | tcp://localhost:... |     2      |  0   |
    +---------+---------------------+------------+------+
    
    


Maintainer
----------

.. code:: ipython3

    from openfed.core import Maintainer
    
    mt = Maintainer(fed_props, network.state_dict(keep_vars=True))
    
    with mt:
        openfed.F.device_alignment()
        openfed.F.count_step(2)
    
    print(mt)


.. parsed-literal::

    [0;34m<OpenFed>[0m [0;35mMaintainer[0m
    +----------------+-----------+-------+
    |      role      | nick_name | pipes |
    +----------------+-----------+-------+
    | openfed_aggregator |   server  |   1   |
    +----------------+-----------+-------+
    


Step
====

.. code:: ipython3

    openfed.api(mt,
                fed_optim,
                5,
                agg_func=openfed.F.paillier_aggregation,
                private_key=private_key)


.. parsed-literal::

    100%|██████████| 5/5 [00:08<00:00,  1.78s/it]

