<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

<div align=center> <img src="https://github.com/FederalLab/OpenFed/raw/main/docs/_static/image/openfed-logo.png" width="300" /> </div>

# OpenFed: A Comprehensive and Versatile Open-Source Federated Learning Framework

![GitHub last commit](https://img.shields.io/github/last-commit/FederalLab/OpenFed) [![Documentation Status](https://readthedocs.org/projects/openfed/badge/?version=latest)](https://openfed.readthedocs.io) [![PyPI](https://img.shields.io/pypi/v/OpenFed)](https://pypi.org/project/OpenFed) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/OpenFed)](https://pypi.org/project/OpenFed) [![badge](https://github.com/FederalLab/OpenFed/workflows/build/badge.svg)](https://github.com/FederalLab/OpenFed/actions) [![codecov](https://codecov.io/gh/FederalLab/OpenFed/branch/main/graph/badge.svg)](https://codecov.io/gh/FederalLab/OpenFed) [![license](https://img.shields.io/github/license/FederalLab/OpenFed.svg)](https://github.com/FederalLab/OpenFed/blob/master/LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2109.07852-red.svg)](https://arxiv.org/abs/2109.07852)

## Introduction

OpenFed is a foundational library for federated learning research and supports many research projects. It reduces the barrier to entry for both researchers and downstream users of Federated Learning by the targeted removal of existing pain points. For researchers, OpenFed provides a framework wherein new methods can be easily implemented and fairly evaluated against an extensive suite of benchmarks. For downstream users, OpenFed allows Federated Learning to be plug and play within different subject-matter contexts, removing the need for deep expertise in Federated Learning.

## Install

PyTorch >= 1.5.1, python>=3.6

**Build latest version from source**:

```shell
git clone https://github.com/FederalLab/OpenFed.git
cd OpenFed
pip install -e .
```

**Stable version**: `pip install openfed`

## Start Federated Learning in an Unprecedented Simple Way

Refer to [here](examples/simulator.ipynb).

## Citation

If you find this project useful in your research, please consider cite:

```bash
@misc{chen2021openfed,
      title={OpenFed: A Comprehensive and Versatile Open-Source Federated Learning Framework},
      author={Dengsheng Chen and Vince Tan and Zhilin Lu and Jie Hu},
      year={2021},
      eprint={2109.07852},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

## Contributing

We appreciate all contributions to improve OpenFed.
Please refer to [CONTRIBUTING.md](https://github.com/FederalLab/OpenFed/raw/main/CONTRIBUTING.md) for the contributing guideline.

## License

OpenFed is released under the MIT License.
