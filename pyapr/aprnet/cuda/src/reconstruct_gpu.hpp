//
// Created by joel on 23.02.23.
//

#ifndef APRNET_RECONSTRUCT_GPU_HPP
#define APRNET_RECONSTRUCT_GPU_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <numerics/miscCuda.hpp>

template<typename scalar_t>
void apply_reconstruct( GPUAccessHelper &access,
						GPUAccessHelper &tree_access,
                        const torch::PackedTensorAccessor32<scalar_t, 3> input,
                        torch::PackedTensorAccessor32<scalar_t, 5> output,
                        const int min_occupied_level,
                        const int batch,
                        const int level_delta);


template<typename scalar_t>
void apply_reconstruct_backward(GPUAccessHelper &access,
                                GPUAccessHelper &tree_access,
                                torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                                const torch::PackedTensorAccessor32<scalar_t, 5> grad_output,
                                const int min_occupied_level,
                                const int batch,
                                const int level_delta);

#endif // APRNET_RECONSTRUCT_GPU_HPP
