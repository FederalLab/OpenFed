### v0.0.0

In order to flexibly support more federated algorithms and projects, like `scaffold`, `mmcv`, the directory of `openfed` might be refactored.

`v0.0.0`'s directory was organized as follows.

```shell
openfed
├── __init__.py
├── api.py
├── common
│   ├── __init__.py
│   ├── address.py
│   └── meta.py
├── core
│   ├── __init__.py
│   ├── const.py
│   ├── functional.py
│   └── maintainer.py
├── data
│   ├── __init__.py
│   ├── datasets.py
│   ├── partitioner.py
│   └── utils.py
├── federated
│   ├── __init__.py
│   ├── const.py
│   ├── exceptions.py
│   ├── functional.py
│   ├── pipe.py
│   └── props.py
├── functional
│   ├── __init__.py
│   ├── agg.py
│   ├── const.py
│   ├── hooks.py
│   ├── paillier.py
│   ├── reduce.py
│   └── step.py
├── optim
│   ├── __init__.py
│   ├── elastic.py
│   ├── fed_optim.py
│   ├── prox.py
│   └── scaffold.py
├── third_party
│   ├── __init__.py
│   ├── fast_nlp
│   │   └── __init__.py
│   ├── mmcv
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── runner
│   │       ├── __init__.py
│   │       └── openfed_runner_constructor.py
│   ├── monai
│   │   └── __init__.py
│   ├── pytorch_lightning
│   │   ├── __init__.py
│   │   └── openfed_checkpoint.py
│   └── stable_baseline3
│       └── __init__.py
├── tools
│   ├── __init__.py
│   ├── simulator.py
│   └── topo_builder.py
├── topo
│   ├── __init__.py
│   ├── functional.py
│   └── topo.py
├── utils
│   ├── __init__.py
│   ├── table.py
│   └── utils.py
└── version.py
```
