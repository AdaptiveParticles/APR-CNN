//
// Created by joel on 28.06.22.
//

#ifndef APRNET_CONV_GPU_HPP
#define APRNET_CONV_GPU_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <numerics/miscCuda.hpp>


#define ACCUMULATE_CONV_333(output, index, z, x, y, neighbour_sum)\
for(int ch_out = 0; ch_out < num_channels_out; ++ch_out) {\
    neighbour_sum = 0;\
    for (int q = 0; q < 3; ++q) {\
        neighbour_sum  += local_stencil[ch_out][q][0][0] * local_patch[z + q - 1][x + 0 - 1][(y+CHUNKSIZE-1)%CHUNKSIZE]\
                        + local_stencil[ch_out][q][0][1] * local_patch[z + q - 1][x + 0 - 1][(y+CHUNKSIZE+0)%CHUNKSIZE]\
                        + local_stencil[ch_out][q][0][2] * local_patch[z + q - 1][x + 0 - 1][(y+CHUNKSIZE+1)%CHUNKSIZE]\
                        + local_stencil[ch_out][q][1][0] * local_patch[z + q - 1][x + 1 - 1][(y+CHUNKSIZE-1)%CHUNKSIZE]\
                        + local_stencil[ch_out][q][1][1] * local_patch[z + q - 1][x + 1 - 1][(y+CHUNKSIZE+0)%CHUNKSIZE]\
                        + local_stencil[ch_out][q][1][2] * local_patch[z + q - 1][x + 1 - 1][(y+CHUNKSIZE+1)%CHUNKSIZE]\
                        + local_stencil[ch_out][q][2][0] * local_patch[z + q - 1][x + 2 - 1][(y+CHUNKSIZE-1)%CHUNKSIZE]\
                        + local_stencil[ch_out][q][2][1] * local_patch[z + q - 1][x + 2 - 1][(y+CHUNKSIZE+0)%CHUNKSIZE]\
                        + local_stencil[ch_out][q][2][2] * local_patch[z + q - 1][x + 2 - 1][(y+CHUNKSIZE+1)%CHUNKSIZE];\
    }\
    atomicAdd(&output[batch][ch_out + channel_offset_out][index], neighbour_sum);\
}\


#define CONV_333_SPREAD_GRAD_INPUT(grad_out_patch, grad_in_patch, z, x, y)\
for(int dz = -1; dz <= 1; ++dz) {\
    for(int dx = -1; dx <= 1; ++dx) {\
        if((z - dz) > 0 && (z - dz) < (BLOCKSIZE - 1) && (x - dx) > 0 && (x - dx) < (BLOCKSIZE - 1)) {\
            for(int dy = -1; dy <= 1; ++dy) {\
                for(int ch_out = 0; ch_out < num_channels_out; ++ch_out) {\
                    auto dO = grad_out_patch[ch_out][z-dz-1][x-dx-1][(y+CHUNKSIZE-dy)%CHUNKSIZE];\
                    for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {\
                        grad_in_patch[ch_in][z][x][y] += dO * local_stencil[ch_out][ch_in][dz+1][dx+1][dy+1];\
                    }\
                }\
            }\
        }\
    }\
}\


#define CONV_333_ACCUMULATE_GRAD_WEIGHTS(grad_out_patch, input_patch, grad_weights_local, grad_weights, dz, dx, dy, stencil_num)\
for(int ch_out = 0; ch_out < num_channels_out; ++ch_out) {\
    for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {\
        grad_weights_local[dz][dx][dy][threadIdx.x] = 0;\
        for(int z_o = 0; z_o < BLOCKSIZE-2; ++z_o) {\
            for(int x_o = 0; x_o < BLOCKSIZE-2; ++x_o) {\
                const auto dO = grad_out_patch[ch_out][z_o][x_o][(threadIdx.x + 1) % CHUNKSIZE];\
                grad_weights_local[dz][dx][dy][threadIdx.x] += dO * input_patch[ch_in][z_o+dz][x_o+dx][(threadIdx.x + dy) % CHUNKSIZE];\
            }\
        }\
        __syncwarp();\
        for(unsigned int s = CHUNKSIZE/2; s > 0; s >>= 1) {\
            if(threadIdx.x < s) {\
                grad_weights_local[dz][dx][dy][threadIdx.x] += grad_weights_local[dz][dx][dy][threadIdx.x + s];\
            }\
            __syncwarp();\
        }\
        if(threadIdx.x == 0) {\
            atomicAdd(&grad_weights[stencil_num][channel_offset_out + ch_out][channel_offset_in + ch_in][dz][dx][dy], grad_weights_local[dz][dx][dy][0]);\
        }\
        __syncwarp();\
    }\
}\


template<typename scalar_t>
void apply_conv333(GPUAccessHelper &access,
                   GPUAccessHelper &tree_access,
                   const torch::PackedTensorAccessor32<scalar_t, 3> input,
                   const torch::PackedTensorAccessor32<scalar_t, 6> weights,
                   torch::PackedTensorAccessor32<scalar_t, 2> tree_data,
                   torch::PackedTensorAccessor32<scalar_t, 3> output,
                   const int batch,
                   const int level_delta);


template<typename scalar_t>
void apply_conv333_backward(GPUAccessHelper &access,
                            GPUAccessHelper &tree_access,
                            torch::PackedTensorAccessor32<scalar_t, 3> input,
                            torch::PackedTensorAccessor32<scalar_t, 6> weights,
                            torch::PackedTensorAccessor32<scalar_t, 2> tree_data,
                            torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                            torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                            torch::PackedTensorAccessor32<scalar_t, 6> grad_weights,
                            torch::PackedTensorAccessor32<scalar_t, 2> grad_tree,
                            const int batch,
                            const int level_delta);



template<typename scalar_t>
void apply_conv111(GPUAccessHelper &access,
                   GPUAccessHelper &tree_access,
                   const torch::PackedTensorAccessor32<scalar_t, 3> input,
                   const torch::PackedTensorAccessor32<scalar_t, 6> weights,
                   torch::PackedTensorAccessor32<scalar_t, 3> output,
                   const int batch,
                   const int level_delta);


template<typename scalar_t>
void apply_conv111_backward(GPUAccessHelper &access,
                            GPUAccessHelper &tree_access,
                            const torch::PackedTensorAccessor32<scalar_t, 3> input,
                            const torch::PackedTensorAccessor32<scalar_t, 6> weights,
                            const torch::PackedTensorAccessor32<scalar_t, 6> weights_transposed,
                            const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                            torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                            torch::PackedTensorAccessor32<scalar_t, 6> grad_weights,
                            const int batch,
                            const int level_delta);

#endif //APRNET_CONV_GPU_HPP
