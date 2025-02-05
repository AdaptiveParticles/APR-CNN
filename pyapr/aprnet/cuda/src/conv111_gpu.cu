#include "conv_gpu.hpp"

#define BLOCKSIZE_X 128
#define CHANNEL_OUT_BLOCKSIZE_FW 4
#define CHANNEL_IN_BLOCKSIZE_FW 32
#define CHANNEL_OUT_BLOCKSIZE_BW 8
#define CHANNEL_IN_BLOCKSIZE_BW 8

template<typename scalar_t>
__global__ void conv111_kernel(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 3> output,
        const torch::PackedTensorAccessor32<scalar_t, 6> weights,
        const int current_max_level,
        const int num_level_deltas,
        const int batch,
        const size_t num_particles) {

    const int level_delta = blockIdx.z % num_level_deltas;
    const int level = current_max_level - level_delta;

    const int channel_offset_in = CHANNEL_IN_BLOCKSIZE_FW * (blockIdx.z / num_level_deltas);
    const int num_channels_in = min(CHANNEL_IN_BLOCKSIZE_FW, input.size(1) - channel_offset_in);
    const int channel_offset_out = CHANNEL_OUT_BLOCKSIZE_FW * blockIdx.y;
    const int num_channels_out = min(CHANNEL_OUT_BLOCKSIZE_FW, output.size(1) - channel_offset_out);


    const size_t level_start = level_delta == (num_level_deltas - 1) ? 0 : xz_end_vec[level_xz_vec[level] - 1];
    const size_t level_end = level < current_max_level ? xz_end_vec[level_xz_vec[level + 1] - 1] : num_particles;

    if(level_start + blockIdx.x * blockDim.x >= level_end) {
        return; // entire block out of range
    }
    
    __shared__ scalar_t local_weights[CHANNEL_IN_BLOCKSIZE_FW];

    for(int ch_out = 0; ch_out < num_channels_out; ++ch_out) {
        __syncthreads();
        if(threadIdx.x < num_channels_in) {
            local_weights[threadIdx.x] = weights[level_delta][channel_offset_out + ch_out][channel_offset_in + threadIdx.x][0][0][0];
        }
        
        __syncthreads();

        for(size_t idx = level_start + threadIdx.x + blockIdx.x * blockDim.x; idx < level_end; idx += blockDim.x * gridDim.x) {

            scalar_t res = 0;
            for(int ch_in = 0; ch_in < num_channels_in; ch_in++) {
                res += local_weights[ch_in] * input[batch][channel_offset_in + ch_in][idx];
            }

            atomicAdd(&output[batch][channel_offset_out + ch_out][idx], res);
        }
    }
}



template<typename scalar_t>
__global__ void conv111_kernel_grad_weights(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
        torch::PackedTensorAccessor32<scalar_t, 6> grad_weights,
        const int current_max_level,
        const int num_level_deltas,
        const int batch,
        const size_t num_particles) {

    const int level_delta = blockIdx.z % num_level_deltas;
    const int level = current_max_level - level_delta;

    const int channel_offset_in = CHANNEL_IN_BLOCKSIZE_BW * (blockIdx.z / num_level_deltas);
    const int num_channels_in = min(CHANNEL_IN_BLOCKSIZE_BW, input.size(1) - channel_offset_in);
    const int channel_offset_out = CHANNEL_OUT_BLOCKSIZE_BW * blockIdx.y;
    const int num_channels_out = min(CHANNEL_OUT_BLOCKSIZE_BW, grad_output.size(1) - channel_offset_out);

    const size_t level_start = (level_delta == (num_level_deltas - 1)) ? 0 : xz_end_vec[level_xz_vec[level] - 1];
    const size_t level_end = (level < current_max_level) ? xz_end_vec[level_xz_vec[level + 1] - 1] : num_particles;

    if(level_start + blockIdx.x * blockDim.x >= level_end) { 
        return; // entire block out of range
    }

    __shared__ scalar_t value_cache[BLOCKSIZE_X];

    for(int ch_out = 0; ch_out < num_channels_out; ++ch_out) {
        for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
            
            __syncthreads();
            value_cache[threadIdx.x] = 0;

            for(size_t idx = level_start + threadIdx.x + blockIdx.x * blockDim.x; idx < level_end; idx += blockDim.x * gridDim.x) {
                value_cache[threadIdx.x] += grad_output[batch][ch_out + channel_offset_out][idx] * input[batch][ch_in + channel_offset_in][idx];
            }

            __syncthreads();

            for(unsigned int s = BLOCKSIZE_X / 2; s > 0; s >>= 1) {
                if(threadIdx.x < s) {
                    value_cache[threadIdx.x] += value_cache[threadIdx.x + s];
                }
                __syncthreads();
            }

            if(threadIdx.x == 0) {
                atomicAdd(&grad_weights[level_delta][ch_out + channel_offset_out][ch_in + channel_offset_in][0][0][0], value_cache[0]);
            }
        }
    }
}


template<typename scalar_t>
void apply_conv111(GPUAccessHelper &access,
                   GPUAccessHelper &tree_access,
                   const torch::PackedTensorAccessor32<scalar_t, 3> input,
                   const torch::PackedTensorAccessor32<scalar_t, 6> weights,
                   torch::PackedTensorAccessor32<scalar_t, 3> output,
                   const int batch,
                   const int level_delta) {
    
    const int num_stencils = weights.size(0);
    const int num_in_channels = input.size(1);
    const int num_out_channels = output.size(1);
    const int current_max_level = access.level_max() - level_delta;

    const size_t num_upgraded_parts = current_max_level < access.level_max() ? \
        tree_access.total_number_particles(current_max_level) - tree_access.total_number_particles(current_max_level - 1) : 0;
    
    const size_t num_particles = access.total_number_particles(current_max_level) + num_upgraded_parts;

    const int num_level_deltas = std::min(num_stencils, current_max_level - 1);
    const int channel_blocks_in = (num_in_channels + CHANNEL_IN_BLOCKSIZE_FW - 1) / CHANNEL_IN_BLOCKSIZE_FW;
    const int channel_blocks_out = (num_out_channels + CHANNEL_OUT_BLOCKSIZE_FW - 1) / CHANNEL_OUT_BLOCKSIZE_FW;

    const int grid_size_x = std::min((num_particles + BLOCKSIZE_X * 128 - 1) / (BLOCKSIZE_X * 128), size_t(2048));

    const dim3 block_dim(BLOCKSIZE_X, 1, 1);
    const dim3 grid_dim(grid_size_x, channel_blocks_out, channel_blocks_in * num_level_deltas);

    conv111_kernel
        <scalar_t>
            <<<grid_dim, block_dim>>>(
                access.get_level_xz_vec_ptr(),
                access.get_xz_end_vec_ptr(),
                input,
                output,
                weights,
                current_max_level,
                num_level_deltas,
                batch,
                num_particles);
}



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
                            const int level_delta) {
    
    const int num_stencils = weights.size(0);
    const int num_in_channels = input.size(1);
    const int num_out_channels = grad_output.size(1);
    const int current_max_level = access.level_max() - level_delta;

    const size_t num_upgraded_parts = current_max_level < access.level_max() ? \
        tree_access.total_number_particles(current_max_level) - tree_access.total_number_particles(current_max_level - 1) : 0;
    
    const size_t num_particles = access.total_number_particles(current_max_level) + num_upgraded_parts;

    const int num_level_deltas = std::min(num_stencils, current_max_level - 1);

    // compute grad_input as the convolution of weights and grad_output
    {
        const int channel_blocks_in = (num_out_channels + CHANNEL_IN_BLOCKSIZE_FW - 1) / CHANNEL_IN_BLOCKSIZE_FW;
        const int channel_blocks_out = (num_in_channels + CHANNEL_OUT_BLOCKSIZE_FW - 1) / CHANNEL_OUT_BLOCKSIZE_FW;

        const int grid_size_x = std::min((num_particles + BLOCKSIZE_X * 128 - 1) / (BLOCKSIZE_X * 128), size_t(2048));

        const dim3 block_dim(BLOCKSIZE_X, 1, 1);
        const dim3 grid_dim(grid_size_x, channel_blocks_out, channel_blocks_in * num_level_deltas);

        conv111_kernel
            <scalar_t>
                <<<grid_dim, block_dim>>>(
                    access.get_level_xz_vec_ptr(),
                    access.get_xz_end_vec_ptr(),
                    grad_output,
                    grad_input,
                    weights_transposed,
                    current_max_level,
                    num_level_deltas,
                    batch,
                    num_particles);
    }

    // compute grad_weights
    {
        const int channel_blocks_in = (num_in_channels + CHANNEL_IN_BLOCKSIZE_BW - 1) / CHANNEL_IN_BLOCKSIZE_BW;
        const int channel_blocks_out = (num_out_channels + CHANNEL_OUT_BLOCKSIZE_BW - 1) / CHANNEL_OUT_BLOCKSIZE_BW;

        const int grid_size_x = std::min((num_particles + BLOCKSIZE_X * 128 - 1) / (BLOCKSIZE_X * 128), size_t(2048));
        
        const dim3 block_dim(BLOCKSIZE_X, 1, 1);
        const dim3 grid_dim(grid_size_x, channel_blocks_out, channel_blocks_in * num_level_deltas);

        conv111_kernel_grad_weights
            <scalar_t>
                <<<grid_dim, block_dim>>>(
                    access.get_level_xz_vec_ptr(),
                    access.get_xz_end_vec_ptr(),
                    input,
                    grad_output,
                    grad_weights,
                    current_max_level,
                    num_level_deltas,
                    batch,
                    num_particles);
    }
}




template void apply_conv111<float>(
        GPUAccessHelper &,
        GPUAccessHelper &,
        const torch::PackedTensorAccessor32<float, 3>,
        const torch::PackedTensorAccessor32<float, 6>,
        torch::PackedTensorAccessor32<float, 3>,
        const int,
        const int);

template void apply_conv111<double>(
        GPUAccessHelper &,
        GPUAccessHelper &,
        const torch::PackedTensorAccessor32<double, 3>,
        const torch::PackedTensorAccessor32<double, 6>,
        torch::PackedTensorAccessor32<double, 3>,
        const int,
        const int);


template void apply_conv111_backward<float>(
        GPUAccessHelper &,
        GPUAccessHelper &,
        const torch::PackedTensorAccessor32<float, 3>,
        const torch::PackedTensorAccessor32<float, 6>,
        const torch::PackedTensorAccessor32<float, 6>,
        const torch::PackedTensorAccessor32<float, 3>,
        torch::PackedTensorAccessor32<float, 3>,
        torch::PackedTensorAccessor32<float, 6>,
        const int,
        const int);

template void apply_conv111_backward<double>(
        GPUAccessHelper &,
        GPUAccessHelper &,
        const torch::PackedTensorAccessor32<double, 3>,
        const torch::PackedTensorAccessor32<double, 6>,
        const torch::PackedTensorAccessor32<double, 6>,
        const torch::PackedTensorAccessor32<double, 3>,
        torch::PackedTensorAccessor32<double, 3>,
        torch::PackedTensorAccessor32<double, 6>,
        const int,
        const int);
