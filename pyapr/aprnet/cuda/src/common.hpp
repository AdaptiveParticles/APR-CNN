//
// Created by joel on 30.06.22.
//

#ifndef PYLIBAPR_COMMON_HPP
#define PYLIBAPR_COMMON_HPP

#include <torch/library.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_LEVEL_DELTA(dlvl, lmax) TORCH_CHECK(lmax > dlvl, "level delta " #dlvl " is too large for APR with maximum level " #lmax)


#endif //PYLIBAPR_COMMON_HPP
