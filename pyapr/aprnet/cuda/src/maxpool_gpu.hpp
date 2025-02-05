//
// Created by Joel Jonsson on 29.06.22.
//

#ifndef APRNET_MAXPOOL_GPU_HPP
#define APRNET_MAXPOOL_GPU_HPP

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <numerics/miscCuda.hpp>
#include <numerics/APRDownsampleGPU.hpp>


template<typename scalar_t>
__global__ void copy_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 3> output,
        const size_t num_elements,
        const int batch);


template<typename scalar_t>
void apply_maxpool(GPUAccessHelper &access,
                    GPUAccessHelper &tree_access,
                    const torch::PackedTensorAccessor32<scalar_t, 3> input,
                    torch::PackedTensorAccessor32<scalar_t, 3> output,
                    torch::PackedTensorAccessor32<int64_t, 3> max_indices,
                    const int batch,
                    const int level_delta);


template<typename scalar_t>
void apply_maxpool_backward(torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                            const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                            const torch::PackedTensorAccessor32<int64_t, 3> max_indices,
                            const size_t num_fixed_particles,
                            const size_t num_tree_particles,
                            const int batch);


#endif //APRNET_MAXPOOL_GPU_HPP
