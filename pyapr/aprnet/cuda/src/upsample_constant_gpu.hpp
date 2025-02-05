
//
// Created by Joel Jonsson on 17.10.22.
//

#ifndef APRNET_UPSAMPLE_CONSTANT_GPU_HPP
#define APRNET_UPSAMPLE_CONSTANT_GPU_HPP

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <numerics/miscCuda.hpp>
#include <numerics/APRDownsampleGPU.hpp>


template<typename scalar_t>
void apply_upsample_const(GPUAccessHelper &access,
                          GPUAccessHelper &tree_access,
                          const torch::PackedTensorAccessor32<scalar_t, 3> input,
                          torch::PackedTensorAccessor32<scalar_t, 3> output,
                          const int batch,
                          const int level_delta);

template<typename scalar_t>
void apply_upsample_const_backward(GPUAccessHelper &access,
                                   GPUAccessHelper &tree_access,
                                   torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                                   const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                                   const int batch,
                                   const int level_delta);

#endif //APRNET_UPSAMPLE_CONSTANT_GPU_HPP
