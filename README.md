<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

<div align=center>
<img src="docs/logo-long.png" width="432" height="183" />
</div>

# OpenFed: An Open-Source AI Safety and Security Guaranteed Deep Learning Framework


![GitHub last commit](https://img.shields.io/github/last-commit/FederalLab/OpenFed) [![Documentation Status](https://readthedocs.org/projects/openfed/badge/?version=latest)](https://openfed.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/OpenFed)](https://pypi.org/project/OpenFed) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/OpenFed)](https://pypi.org/project/OpenFed)

## Introduction

OpenFed is a foundational library for federated learning research and supports many research projects as below:

- [benchmark-lightly](https://github.com/FederalLab/benchmark-lightly): FederalLab's simulation benchmark.
- [openfed-cv](https://github.com/FederalLab/openfed-cv): FederalLab's toolkit and benchmark for computer vision in federated learning. This toolkit is based on [mmcv](https://github.com/open-mmlab/mmcv/), and provides the federated learning for following tasks:
  - [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
  - [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
  - [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
  - [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
  - [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
  - [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
  - [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
  - [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
  - [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition and understanding toolbox.
  - [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [openfed-finance](https://github.com/FederalLab/openfed-finance): FederalLab's toolbox and benchmark for finance data analysis in federated learning.
- [openfed-medical](https://github.com/FederalLab/openfed-medical): FederalLab's toolbox and benchmark for medical data analysis in federated learning. It is based on [MONAI](https://github.com/Project-MONAI/MONAI).
- [openfed-nlp](https://github.com/FederalLab/openfed-nlp): FederalLab's toolbox and benchmark for natural language processing in federated learning. It is based on [transformers](https://github.com/huggingface/transformers).
- [openfed-rl](https://github.com/FederalLab/openfed-rl): FederalLab's toolbox and benchmark for reinforcement learning in federated learning. It is based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

In addition, we also provide a toolkit for better compatibility with following libraries, so that you can use OpenFed with those libraries without obstacles and more easily:

- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning): The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.
- [mmcv](https://github.com/open-mmlab/mmcv): MMCV is a foundational library for computer vision research and supports many research projects.

## Install

Python>=3.7, PyTorch>=1.8.0 are required.

**Stable version**: `pip install openfed`

**Latest version**: `pip install -e git+https://github.com/FederalLab/OpenFed.git`


## Citation

If you find this project useful in your research, please consider cite:

```
@misc{OpenFed,
Author = {Chen Dengsheng},
Title = {OpenFed: An Open-Source Security and Privacy Guaranteed Federated Learning Framework},
Year = {2021},
Eprint = {arXiv:2109.07852},
}
```
