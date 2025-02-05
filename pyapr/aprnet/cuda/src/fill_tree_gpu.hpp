//
// Created by joel on 26.08.22.
//

#ifndef APRNET_FILL_TREE_GPU_HPP
#define APRNET_FILL_TREE_GPU_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <numerics/miscCuda.hpp>
#include <numerics/APRDownsampleGPU.hpp>


template<typename scalar_t>
void apply_fill_tree_mean(
        GPUAccessHelper &access,
        GPUAccessHelper &tree_access,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 2> output,
        const int batch,
        const int level_delta);


template<typename scalar_t>
void apply_fill_tree_mean_backward(
        GPUAccessHelper &access,
        GPUAccessHelper &tree_access,
        torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
        torch::PackedTensorAccessor32<scalar_t, 2> grad_tree,
        const int batch,
        const int level_delta);

#endif //APRNET_FILL_TREE_GPU_HPP
