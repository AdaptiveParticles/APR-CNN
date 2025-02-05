//
// Created by joel on 24.08.22.
//

#ifndef APRNET_RESTRICT_GPU_HPP
#define APRNET_RESTRICT_GPU_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <numerics/miscCuda.hpp>


template<typename scalar_t>
__global__ void copy_weights(const torch::PackedTensorAccessor32<scalar_t, 6> weights_in,
                               torch::PackedTensorAccessor32<scalar_t, 6> weights_out);


template<typename scalar_t>
__global__ void restrict_kernel_333(const torch::PackedTensorAccessor32<scalar_t, 6> weights_in,
                                    torch::PackedTensorAccessor32<scalar_t, 6> weights_out,
                                    const int max_channels);


template<typename scalar_t>
void apply_restrict_kernel_333(const torch::PackedTensorAccessor32<scalar_t, 6> weights_in,
                               torch::PackedTensorAccessor32<scalar_t, 6> weights_out);


template<typename scalar_t>
void apply_restrict_kernel_333_backward(torch::PackedTensorAccessor32<scalar_t, 6> grad_weights_in,
                                        const torch::PackedTensorAccessor32<scalar_t, 6> grad_weights_out);

#endif //APRNET_RESTRICT_GPU_HPP