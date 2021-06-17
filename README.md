# OpenFed

## 创建一个只有单一客户端和服务器的联邦学习环境

![Demo](doc/demo.png)


## 文件结构

```bash
    openfed
    ├── __init__.py
    ├── aggregate
    │   ├── __init__.py
    │   ├── aggregator.py
    │   ├── average.py
    │   ├── elastic.py
    │   ├── naive.py
    │   └── readme.md
    ├── backend.py
    ├── common
    │   ├── __init__.py
    │   ├── address.py
    │   ├── constants.py
    │   ├── hook.py
    │   ├── package.py
    │   ├── readme.md
    │   ├── thread.py
    │   ├── vars.py
    │   └── wrapper.py
    ├── data
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── nlp
    │   │   ├── __init__.py
    │   │   ├── shakespear.py
    │   │   └── stackoverflow.py
    │   ├── partitioner.py
    │   ├── readme.md
    │   └── vision
    │       ├── __init__.py
    │       └── emnist.py
    ├── federated
    │   ├── __init__.py
    │   ├── core.py
    │   ├── deliver
    │   │   ├── __init__.py
    │   │   ├── delivery.py
    │   │   ├── functional.py
    │   │   └── readme.md
    │   ├── federated.py
    │   ├── inform
    │   │   ├── __init__.py
    │   │   ├── functional.py
    │   │   ├── informer.py
    │   │   └── readme.md
    │   └── readme.md
    ├── frontend.py
    ├── launch.py
    ├── optim
    │   ├── __init__.py
    │   ├── elastic_aux.py
    │   └── readme.md
    ├── readme.md
    └── utils
        ├── __init__.py
        ├── helper.py
        ├── readme.md
        └── utils.py

    10 directories, 48 files
```