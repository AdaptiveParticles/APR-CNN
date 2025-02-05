# APR-CNN

This repository contains the code for the paper:

- Joel Jonsson, Bevan L. Cheeseman, and Ivo F. Sbalzarini. Convolutional Neural Networks for the Adaptive Particle Representation of Large Microscopy Images. Transactions on Machine Learning Research (TMLR), 2025.

Please cite the paper if you use the code.


# Installation

The build is experimental and may require some tweaking. In principle, it should be possible to build the package as follows (assuming an A100 GPU):

```bash
torch_path="$(python -c 'import torch.utils; print(torch.utils.cmake_pre
fix_path)')"

EXTRA_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$torch_path -DTORCH_CUDA_ARCH_LIST='8.0' -DCMAKE_CUDA_ARCHITECTURES=80" pip install . -v
```

The `apr_models` package can then be installed as follows:

```bash
pip install ./apr_models
```
