# Build OpenFed from source

## Build on Linux or macOS

```bash
git clone https://github.com/FederalLab/OpenFed.git
cd OpenFed
pip install -e .
```

## Build on Windows

Building OpenFed on Windows is a familiar with that on Linux.

## Test

```shell
(openfed) ./pytest.sh
General test...
================= test session starts ==================
platform darwin -- Python 3.7.10, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: /Users/densechen/code/OpenFed
plugins: xdist-2.4.0, ordering-0.6, forked-1.3.0
collected 32 items / 11 deselected / 21 selected

tests/test_simulator.py .                        [  4%]
tests/test_data/test_partitioner.py .            [  9%]
tests/test_api.py .                              [ 14%]
tests/test_build.py .                            [ 19%]
tests/test_common/test_address.py ....           [ 38%]
tests/test_common/test_meta.py .                 [ 42%]
tests/test_data/test_partitioner.py ...          [ 57%]
tests/test_topo/test_topo.py ....                [ 76%]
tests/test_utils/test_table.py ..                [ 85%]
tests/test_utils/test_utils.py ...               [100%]

========== 21 passed, 11 deselected in 1.43s ===========
Federated...
================= test session starts ==================
platform darwin -- Python 3.7.10, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: /Users/densechen/code/OpenFed
plugins: xdist-2.4.0, ordering-0.6, forked-1.3.0
gw0 [3] / gw1 [3] / gw2 [3]
...                                              [100%]
================== 3 passed in 3.80s ===================
Maintainer...
================= test session starts ==================
platform darwin -- Python 3.7.10, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: /Users/densechen/code/OpenFed
plugins: xdist-2.4.0, ordering-0.6, forked-1.3.0
gw0 [3] / gw1 [3] / gw2 [3]
...                                              [100%]
================== 3 passed in 1.73s ===================
Simulator...
================= test session starts ==================
platform darwin -- Python 3.7.10, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: /Users/densechen/code/OpenFed
plugins: xdist-2.4.0, ordering-0.6, forked-1.3.0
gw0 [4] / gw1 [4] / gw2 [4]
....                                             [100%]
================== 4 passed in 1.84s ===================
Paillier Crypt...
================= test session starts ==================
platform darwin -- Python 3.7.10, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: /Users/densechen/code/OpenFed
plugins: xdist-2.4.0, ordering-0.6, forked-1.3.0
gw0 [2] / gw1 [2]
100%|█████████████████████| 2/2 [00:03<00:00,  1.87s/it]
.                                               [100%]
================== 2 passed in 5.28s ===================
(openfed)  densechen@C02DW0CQMD6R  ~/code/OpenFed   main ± 
```
