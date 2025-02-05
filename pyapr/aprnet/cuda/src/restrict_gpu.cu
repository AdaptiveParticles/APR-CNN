#include "restrict_gpu.hpp"

#define MAX_CHANNELS 32

template<typename scalar_t>
__global__ void copy_weights(const torch::PackedTensorAccessor32<scalar_t, 6> weights_in,
                             torch::PackedTensorAccessor32<scalar_t, 6> weights_out) {

    const size_t num_elements = weights_in.stride(0) * weights_in.size(0);

    for(size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
        const size_t level = idx / weights_in.stride(0);
        size_t tmp = idx - level * weights_in.stride(0);
        const size_t ch_out = tmp / weights_in.stride(1);
        tmp -= ch_out * weights_in.stride(1);
        const size_t ch_in = tmp / weights_in.stride(2);
        tmp -= ch_in * weights_in.stride(2);
        const size_t z = tmp / weights_in.stride(3);
        tmp -= z * weights_in.stride(3);
        const size_t x = tmp / weights_in.stride(4);
        const size_t y = tmp - x * weights_in.stride(4);

        weights_out[level][ch_out][ch_in][z][x][y] = weights_in[level][ch_out][ch_in][z][x][y];
    }
}


template<typename scalar_t>
__global__ void copy_weights_backward(torch::PackedTensorAccessor32<scalar_t, 6> grad_weights_in,
                                      const torch::PackedTensorAccessor32<scalar_t, 6> grad_weights_out) {

    const size_t num_elements = grad_weights_in.stride(0) * grad_weights_in.size(0);

    for(size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
        const size_t level = idx / grad_weights_in.stride(0);
        size_t tmp = idx - level * grad_weights_in.stride(0);
        const size_t ch_out = tmp / grad_weights_in.stride(1);
        tmp -= ch_out * grad_weights_in.stride(1);
        const size_t ch_in = tmp / grad_weights_in.stride(2);
        tmp -= ch_in * grad_weights_in.stride(2);
        const size_t z = tmp / grad_weights_in.stride(3);
        tmp -= z * grad_weights_in.stride(3);
        const size_t x = tmp / grad_weights_in.stride(4);
        const size_t y = tmp - x * grad_weights_in.stride(4);

        grad_weights_in[level][ch_out][ch_in][z][x][y] = grad_weights_out[level][ch_out][ch_in][z][x][y];
    }
}


template<typename scalar_t>
__global__ void restrict_kernel_333(const torch::PackedTensorAccessor32<scalar_t, 6> weights_in,
                                    torch::PackedTensorAccessor32<scalar_t, 6> weights_out) {
    
    const int source_level = weights_in.size(0) - 1;
    const int level_delta = blockIdx.z + 1;
    const int target_level = weights_in.size(0) - 1 + level_delta;

    const int channel_offset_out = blockIdx.x * MAX_CHANNELS;
    const int channel_offset_in  = blockIdx.y * MAX_CHANNELS;

    const int num_out_channels = min(MAX_CHANNELS, weights_in.size(1) - channel_offset_out);
    const int num_in_channels  = min(MAX_CHANNELS, weights_in.size(2) - channel_offset_in);

    const int step_size = (int) pow(2.0f, (float) level_delta);
    const scalar_t factor = std::pow((scalar_t) step_size, scalar_t(3));

    const int z_out = threadIdx.z;
    const int x_out = threadIdx.y;
    
    for(int z_in = 0; z_in < 3; ++z_in) {
        
        const int z_start = max(z_out * step_size, step_size - 1 + z_in);
        const int z_end = min((z_out + 1) * step_size, 2 * step_size - 1 + z_in);
        const scalar_t overlap_z = z_end - z_start;

        if(overlap_z <= 0) { continue; }

        for(int x_in = 0; x_in < 3; ++x_in) {
            
            const int x_start = max(x_out * step_size, step_size - 1 + x_in);
            const int x_end = min((x_out + 1) * step_size, 2 * step_size - 1 + x_in);
            const scalar_t overlap_x = x_end - x_start;

            if(overlap_x <= 0) { continue; }

            for(int y_in = 0; y_in < 3; ++y_in) {
                for(int y_out = 0; y_out < 3; ++y_out) {

                    const int y_start = max(y_out * step_size, step_size - 1 + y_in);
                    const int y_end = min((y_out + 1) * step_size, 2 * step_size - 1 + y_in);
                    const scalar_t overlap_y = y_end - y_start;

                    if(overlap_y <= 0) { continue; }

                    const scalar_t overlap = overlap_x * overlap_y * overlap_z / factor;

                    for(int ch_idx = threadIdx.x; ch_idx < num_out_channels * num_in_channels; ch_idx += blockDim.x) {
                        const int ch_out = ch_idx / num_in_channels;
                        const int ch_in = ch_idx - ch_out * num_in_channels;

                        weights_out[target_level][channel_offset_out + ch_out][channel_offset_in + ch_in][z_out][x_out][y_out] += 
                            overlap * weights_in[source_level][channel_offset_out + ch_out][channel_offset_in + ch_in][z_in][x_in][y_in];
                    }
                }
            }
        }
    }
}



template<typename scalar_t>
__global__ void restrict_kernel_333_backward(torch::PackedTensorAccessor32<scalar_t, 6> grad_weights_in,
                                             const torch::PackedTensorAccessor32<scalar_t, 6> grad_weights_out) {
    
    const int source_level = grad_weights_in.size(0) - 1;
    const int level_delta = blockIdx.z + 1;
    const int target_level = grad_weights_in.size(0) - 1 + level_delta;

    const int channel_offset_out = blockIdx.x * MAX_CHANNELS;
    const int channel_offset_in  = blockIdx.y * MAX_CHANNELS;

    const int num_out_channels = min(MAX_CHANNELS, grad_weights_in.size(1) - channel_offset_out);
    const int num_in_channels  = min(MAX_CHANNELS, grad_weights_in.size(2) - channel_offset_in);

    const int step_size = (int) pow(2.0f, (float) level_delta);
    const scalar_t factor = std::pow((scalar_t) step_size, scalar_t(3));

    const int z_out = threadIdx.z;
    const int x_out = threadIdx.y;
    
    for(int z_in = 0; z_in < 3; ++z_in) {
        
        const int z_start = max(z_out * step_size, step_size - 1 + z_in);
        const int z_end = min((z_out + 1) * step_size, 2 * step_size - 1 + z_in);
        const scalar_t overlap_z = z_end - z_start;

        if(overlap_z <= 0) { continue; }

        for(int x_in = 0; x_in < 3; ++x_in) {
            
            const int x_start = max(x_out * step_size, step_size - 1 + x_in);
            const int x_end = min((x_out + 1) * step_size, 2 * step_size - 1 + x_in);
            const scalar_t overlap_x = x_end - x_start;

            if(overlap_x <= 0) { continue; }

            for(int y_in = 0; y_in < 3; ++y_in) {
                for(int y_out = 0; y_out < 3; ++y_out) {

                    const int y_start = max(y_out * step_size, step_size - 1 + y_in);
                    const int y_end = min((y_out + 1) * step_size, 2 * step_size - 1 + y_in);
                    const scalar_t overlap_y = y_end - y_start;

                    if(overlap_y <= 0) { continue; }

                    const scalar_t overlap = overlap_x * overlap_y * overlap_z / factor;

                    for(int ch_idx = threadIdx.x; ch_idx < num_out_channels * num_in_channels; ch_idx += blockDim.x) {
                        const int ch_out = ch_idx / num_in_channels;
                        const int ch_in = ch_idx - ch_out * num_in_channels;
                        
                        atomicAdd(&grad_weights_in[source_level][channel_offset_out + ch_out][channel_offset_in + ch_in][z_in][x_in][y_in],
                                  overlap * grad_weights_out[target_level][channel_offset_out + ch_out][channel_offset_in + ch_in][z_out][x_out][y_out]);
                    }
                }
            }
        }
    }
}



template<typename scalar_t>
void apply_restrict_kernel_333(const torch::PackedTensorAccessor32<scalar_t, 6> weights_in,
                               torch::PackedTensorAccessor32<scalar_t, 6> weights_out) {

    const int num_stencils_in = weights_in.size(0);
    const int num_stencils_out = weights_out.size(0);

    int block_dim, grid_dim, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_dim, copy_weights<scalar_t>, 0, 0);
    grid_dim = std::min((weights_in.stride(0) * weights_in.size(0) + block_dim - 1) / block_dim, min_grid_size);

    copy_weights<scalar_t>
        <<<grid_dim, block_dim>>>
            (weights_in, weights_out);

    const int num_in_blocks  = (weights_in.size(2) + MAX_CHANNELS - 1) / MAX_CHANNELS;
    const int num_out_blocks = (weights_in.size(1) + MAX_CHANNELS - 1) / MAX_CHANNELS;
    const int num_level_blocks = weights_out.size(0) - weights_in.size(0);

    if(num_level_blocks > 0) {
        dim3 blocks_l(num_out_blocks, num_in_blocks, num_level_blocks);
        dim3 threads_l(32, 3, 3);

        restrict_kernel_333<scalar_t>
            <<<blocks_l, threads_l>>>
                (weights_in, weights_out);
    }
    cudaDeviceSynchronize();
}



template<typename scalar_t>
void apply_restrict_kernel_333_backward(torch::PackedTensorAccessor32<scalar_t, 6> grad_weights_in,
                                        const torch::PackedTensorAccessor32<scalar_t, 6> grad_weights_out) {

    const int num_stencils_in = grad_weights_in.size(0);
    const int num_stencils_out = grad_weights_out.size(0);

    int block_dim, grid_dim, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_dim, copy_weights_backward<scalar_t>, 0, 0);
    grid_dim = std::min((grad_weights_in.stride(0) * grad_weights_in.size(0) + block_dim - 1) / block_dim, min_grid_size);

    copy_weights_backward<scalar_t>
        <<<grid_dim, block_dim>>>
            (grad_weights_in, grad_weights_out);

    const int num_in_blocks  = (grad_weights_in.size(2) + MAX_CHANNELS - 1) / MAX_CHANNELS;
    const int num_out_blocks = (grad_weights_in.size(1) + MAX_CHANNELS - 1) / MAX_CHANNELS;
    const int num_level_blocks = grad_weights_out.size(0) - grad_weights_in.size(0);

    if(num_level_blocks > 0) {
        dim3 blocks_l(num_out_blocks, num_in_blocks, num_level_blocks);
        dim3 threads_l(32, 3, 3);

        restrict_kernel_333_backward<scalar_t>
            <<<blocks_l, threads_l>>>
                (grad_weights_in, grad_weights_out);
    }
    cudaDeviceSynchronize();
}


template __global__ void copy_weights<float>(const torch::PackedTensorAccessor32<float, 6>, torch::PackedTensorAccessor32<float, 6>);
template __global__ void copy_weights<double>(const torch::PackedTensorAccessor32<double, 6>, torch::PackedTensorAccessor32<double, 6>);

template __global__ void restrict_kernel_333<float>(const torch::PackedTensorAccessor32<float, 6>, torch::PackedTensorAccessor32<float, 6>);
template __global__ void restrict_kernel_333<double>(const torch::PackedTensorAccessor32<double, 6>, torch::PackedTensorAccessor32<double, 6>);

template void apply_restrict_kernel_333<float>(const torch::PackedTensorAccessor32<float, 6>, torch::PackedTensorAccessor32<float, 6>);
template void apply_restrict_kernel_333<double>(const torch::PackedTensorAccessor32<double, 6>, torch::PackedTensorAccessor32<double, 6>);

template void apply_restrict_kernel_333_backward<float>(torch::PackedTensorAccessor32<float, 6>, const torch::PackedTensorAccessor32<float, 6>);
template void apply_restrict_kernel_333_backward<double>(torch::PackedTensorAccessor32<double, 6>, const torch::PackedTensorAccessor32<double, 6>);
