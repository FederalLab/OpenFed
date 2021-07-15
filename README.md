# OpenFed

## Features

1. Async/Sync Transfer Support
2. Dynamic Address Management
3. Unified Frontend and Backend API
4. Arbitrary Federated Topology Connection
5. Transfer Data without Hesitation
6. PyTorch Coding Style

## Install

Python>=3.7, PyTorch>=1.8.0 are required.

```bash
pip install openfed # not the latest version.
```

```bash
conda create -n openfed python=3.7 -y
conda activate openfed
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts -y

pip3 install -r requirements.txt

# test
python3 -m openfed.tools.launch --nproc_per_node 3 --logdir /tmp tests/cpu.py

# make sure /tmp/openfed.sharefile does not exist before run this script.
python3 -m openfed.tools.launch --nproc_per_node 11 --logdir /tmp tests/test.py --fed_init_method file:///tmp/openfed.sharefile
```

## Docs

```bash
# install requirements
pip install Sphinx
pip install sphinx_rtd_theme

# install openfed to your python path
python setup.py install

cd docs
# generate rst files
sphinx-apidoc -o . ../openfed/

# make html
make html

# docs have been generated under 'docs/_build/html'
```
