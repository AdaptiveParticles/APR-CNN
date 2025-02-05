//
// Created by Joel Jonsson on 29.06.22.
//

#include "maxpool_gpu.hpp"

#define MAX_CHANNELS 32

template<typename scalar_t>
__global__ void copy_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 3> output,
        const size_t num_elements,
        const int batch) {

    for(int channel = 0; channel < input.size(1); ++channel) {
        for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
            output[batch][channel][idx] = input[batch][channel][idx];
        }
    }
}


template<typename scalar_t>
__global__ void unpool_gradient(
        torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
        const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
        const torch::PackedTensorAccessor32<int64_t, 3> max_indices,
        const size_t start_index,
        const size_t num_elements,
        const int batch) {

    for(int channel = 0; channel < grad_input.size(1); ++channel) {
        for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
            const int64_t m_idx = max_indices[batch][channel][start_index + idx];
            grad_input[batch][channel][m_idx] = grad_output[batch][channel][start_index + idx];
        }
    }
}



template<typename scalar_t>
__global__ void max_pool_level_max(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 3> output,
        torch::PackedTensorAccessor32<int64_t, 3> max_indices,
        const int z_num,
        const int x_num,
        const int x_num_parent,
        const int level,
        const int batch,
        const int64_t tree_offset,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset = blockIdx.y * MAX_CHANNELS;
    const int num_channels = min(MAX_CHANNELS, input.size(1) - channel_offset);

    const int z_p = index / x_num_parent;
    const int x_p = index - z_p*x_num_parent;

    //Local identifiers.
    const int x_index = 2 * x_p + threadIdx.x / 64;
    const int z_index = 2 * z_p + (threadIdx.x / 32) % 2;

    const int block = threadIdx.x / 32;
    const int local_th = threadIdx.x % 32;

    //Particles
    __shared__ std::size_t global_index_begin_0[4];
    __shared__ std::size_t global_index_end_0[4];

    //Parent Tree Particle Cells
    __shared__ std::size_t global_index_begin_p[4];
    __shared__ std::size_t global_index_end_p[4];

    //shared memory cache
    __shared__ scalar_t value_cache[MAX_CHANNELS][8][16];
    __shared__ int64_t index_cache[8][16];

    const bool row_within_bounds = x_index < x_num && z_index < z_num;

    if((local_th==0) ) {
        // APR particle sparse row index range
        if(row_within_bounds) {
            size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
            global_index_begin_0[block] = xz_end_vec[xz_start - 1];
            global_index_end_0[block] = xz_end_vec[xz_start];
        }

        // Parent particle sparse row index range
        size_t xz_start = x_p + z_p * x_num_parent + level_xz_vec_tree[level-1];
        global_index_begin_p[block] = xz_end_vec_tree[xz_start - 1];
        global_index_end_p[block] = xz_end_vec_tree[xz_start];
    }

    __syncwarp();

    int y_0, y_p;

    size_t particle_index = global_index_begin_0[block] + local_th;
    size_t parent_index = global_index_begin_p[block] + local_th;

    if (row_within_bounds && particle_index < global_index_end_0[block]) {
        y_0 =  y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }
    __syncwarp();

    if (parent_index < global_index_end_p[block]) {
        y_p = y_vec_tree[parent_index];
    } else{
        y_p = INT32_MAX;
    }
    __syncwarp();

    const int block_start = y_vec_tree[global_index_begin_p[block]] / 16;
    const int block_end = y_vec_tree[global_index_end_p[block] - 1] / 16 + 1;

    for (int y_block = block_start; y_block < block_end; ++y_block) {

        __syncthreads();

        // reset local caches
        index_cache[2 * block + local_th % 2][local_th / 2] = -1;
        for(int ch = 0; ch < num_channels; ++ch) {
            value_cache[ch][2 * block + local_th % 2][local_th / 2] = -1e17;
        }
        __syncwarp();

        // update particle index
        while (y_0 < y_block * 32) {
            particle_index += 32;
            if (particle_index < global_index_end_0[block]) {
                y_0 = y_vec[particle_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent index
        while (y_p < y_block * 16) {
            parent_index += 32;
            if (parent_index < global_index_end_p[block]) {
                y_p = y_vec_tree[parent_index];
            } else {
                y_p = INT32_MAX;
            }
        }
        __syncthreads();

        // update local caches
        if (y_0 < (y_block + 1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                value_cache[ch][2 * block + y_0 % 2][(y_0 / 2) % 16] = input[batch][ch + channel_offset][particle_index];
            }
            index_cache[2 * block + y_0 % 2][(y_0 / 2) % 16] = particle_index;
        }
        __syncthreads();

        // apply downsampling
        if (y_p < (y_block+1) * 16) {
            for(int ch = block; ch < num_channels; ch += 4) {

                scalar_t max_value = value_cache[ch][0][y_p % 16];
                int64_t max_index = index_cache[0][y_p % 16];

                for(int i = 1; i < 8; ++i) {
                    if(value_cache[ch][i][y_p % 16] > max_value) {
                        max_value = value_cache[ch][i][y_p % 16];
                        max_index = index_cache[i][y_p % 16];
                    }
                }
                output[batch][ch + channel_offset][parent_index + tree_offset] = max_value;
                max_indices[batch][ch + channel_offset][parent_index + tree_offset] = max_index;
            }
        }
    }
}



template<typename scalar_t>
__global__ void max_pool_ds(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 3> output,
        torch::PackedTensorAccessor32<int64_t, 3> max_indices,
        const int z_num,
        const int x_num,
        const int x_num_parent,
        const int level,
        const int batch,
        const int64_t tree_offset_in,
        const int64_t tree_offset_out,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset = blockIdx.y * MAX_CHANNELS;
    const int num_channels = min(MAX_CHANNELS, input.size(1) - channel_offset);

    const int z_p = index / x_num_parent;
    const int x_p = index - z_p*x_num_parent;

    // Local identifiers.
    const int x_index = 2 * x_p + threadIdx.x / 64;
    const int z_index = 2 * z_p + (threadIdx.x / 32) % 2;

    const int block = threadIdx.x / 32;
    const int local_th = threadIdx.x % 32;

    // Particles
    __shared__ std::size_t global_index_begin_0[4];
    __shared__ std::size_t global_index_end_0[4];

    //Tree particles
    __shared__ std::size_t global_index_begin_t[4];
    __shared__ std::size_t global_index_end_t[4];

    //Parent Tree Particle Cells
    __shared__ std::size_t global_index_begin_p[4];
    __shared__ std::size_t global_index_end_p[4];

    // shared memory cache
    __shared__ scalar_t value_cache[MAX_CHANNELS][8][16];
    __shared__ int64_t index_cache[8][16];

    const bool row_within_bounds = x_index < x_num && z_index < z_num;

    if( (local_th==0) ) {
        if(row_within_bounds) {
            // APR particle sparse row index range
            size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
            global_index_begin_0[block] = xz_end_vec[xz_start - 1];
            global_index_end_0[block] = xz_end_vec[xz_start];

            // Tree particle sparse row index range
            xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
            global_index_begin_t[block] = xz_end_vec_tree[xz_start - 1];
            global_index_end_t[block] = xz_end_vec_tree[xz_start];
        }

        // Parent particle sparse row index range
        size_t xz_start = x_p + z_p * x_num_parent + level_xz_vec_tree[level-1];
        global_index_begin_p[block] = xz_end_vec_tree[xz_start - 1];
        global_index_end_p[block] = xz_end_vec_tree[xz_start];
    }
    __syncwarp();

    int y_0, y_t, y_p;

    size_t particle_index = global_index_begin_0[block] + local_th;
    size_t tree_index = global_index_begin_t[block] + local_th;
    size_t parent_index = global_index_begin_p[block] + local_th;

    if (row_within_bounds && particle_index < global_index_end_0[block]) {
        y_0 =  y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }
    __syncwarp();

    if (row_within_bounds && tree_index < global_index_end_t[block]) {
        y_t = y_vec_tree[tree_index];
    } else {
        y_t = INT32_MAX;
    }
    __syncwarp();

    if (parent_index < global_index_end_p[block]) {
        y_p = y_vec_tree[parent_index];
    } else {
        y_p = INT32_MAX;
    }
    __syncwarp();

    const int block_start = y_vec_tree[global_index_begin_p[block]] / 16;
    const int block_end = y_vec_tree[global_index_end_p[block] - 1] / 16 + 1;

    for (int y_block = block_start; y_block < block_end; ++y_block) {

        __syncthreads();

        // reset local caches
        index_cache[2 * block + local_th % 2][local_th / 2] = -1;
        for(int ch = 0; ch < num_channels; ++ch) {
            value_cache[ch][2 * block + local_th % 2][local_th / 2] = -1e17;
        }
        __syncwarp();

        // update particle index
        while (y_0 < y_block * 32) {
            particle_index += 32;
            if (particle_index < global_index_end_0[block]) {
                y_0 = y_vec[particle_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update tree particle index
        while (y_t < y_block * 32) {
            tree_index += 32;
            if (tree_index < global_index_end_t[block]) {
                y_t = y_vec_tree[tree_index];
            } else {
                y_t = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent index
        while (y_p < y_block * 16) {
            parent_index += 32;
            if (parent_index < global_index_end_p[block]) {
                y_p = y_vec_tree[parent_index];
            } else {
                y_p = INT32_MAX;
            }
        }
        __syncthreads();

        // update local caches
        if (y_0 < (y_block + 1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                value_cache[ch][2 * block + y_0 % 2][(y_0 / 2) % 16] = input[batch][ch + channel_offset][particle_index];
            }
            index_cache[2 * block + y_0 % 2][(y_0 / 2) % 16] = particle_index;
        }

        if (y_t < (y_block + 1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                value_cache[ch][2 * block + y_t % 2][(y_t / 2) % 16] = input[batch][ch + channel_offset][tree_index + tree_offset_in];
            }
            index_cache[2 * block + y_t % 2][(y_t / 2) % 16] = tree_index + tree_offset_in;
        }

        __syncthreads();

        // apply downsampling
        if (y_p < (y_block + 1) * 16) {
            for(int ch = block; ch < num_channels; ch += 4) {

                scalar_t max_value = value_cache[ch][0][y_p % 16];
                int64_t max_index = index_cache[0][y_p % 16];

                for(int i = 1; i < 8; ++i) {
                    if(value_cache[ch][i][y_p % 16] > max_value) {
                        max_value = value_cache[ch][i][y_p % 16];
                        max_index = index_cache[i][y_p % 16];
                    }
                }
                output[batch][ch + channel_offset][parent_index + tree_offset_out] = max_value;
                max_indices[batch][ch + channel_offset][parent_index + tree_offset_out] = max_index;
            }
        }
    }
}


template<typename scalar_t>
void apply_maxpool( GPUAccessHelper &access,
                    GPUAccessHelper &tree_access,
                    const torch::PackedTensorAccessor32<scalar_t, 3> input,
                    torch::PackedTensorAccessor32<scalar_t, 3> output,
                    torch::PackedTensorAccessor32<int64_t, 3> max_indices,
                    const int batch,
                    const int level_delta) {

    VectorData<int> ne_counter;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu;
    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter, ne_rows_gpu);

    const int current_max_level = access.level_max() - level_delta;
    const int num_channels = input.size(1);
    const int64_t tree_offset_in = access.total_number_particles(current_max_level) - (int64_t) tree_access.total_number_particles(current_max_level - 1);
    const int64_t tree_offset_out = access.total_number_particles(current_max_level - 1) - (int64_t) tree_access.total_number_particles(current_max_level - 2);

    // copy particles up to current_level_max
    const size_t parts_to_copy = access.total_number_particles(current_max_level - 1);

    if(parts_to_copy > 0) {
        int block_dim, grid_dim, min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_dim, copy_kernel<scalar_t>, 0, 0);
        grid_dim = std::min((parts_to_copy + block_dim - 1) / block_dim, (size_t) min_grid_size);

        copy_kernel<scalar_t>
                <<<grid_dim, block_dim>>>(input, output, parts_to_copy, batch);
    }

    // downsample max level
    size_t ne_sz = ne_counter[current_max_level + 1] - ne_counter[current_max_level];
    size_t offset = ne_counter[current_max_level];

    if( ne_sz == 0 ) {
        return; // no particles to downsample
    }

    const int num_channel_blocks = (num_channels + MAX_CHANNELS - 1) / MAX_CHANNELS;

    dim3 grid_dim_l(ne_sz, num_channel_blocks);
    int block_dim = 128;

    if (level_delta > 0) {
        // downsampled version with tree values

        max_pool_ds
            <scalar_t>
                <<<grid_dim_l, block_dim>>>
                    (access.get_level_xz_vec_ptr(),
                     access.get_xz_end_vec_ptr(),
                     access.get_y_vec_ptr(),
                     tree_access.get_level_xz_vec_ptr(),
                     tree_access.get_xz_end_vec_ptr(),
                     tree_access.get_y_vec_ptr(),
                     input,
                     output,
                     max_indices,
                     access.z_num(current_max_level),
                     access.x_num(current_max_level),
                     tree_access.x_num(current_max_level - 1),
                     current_max_level,
                     batch,
                     tree_offset_in,
                     tree_offset_out,
                     ne_rows_gpu.get() + offset);

    } else {
        // simpler max level kernel

        max_pool_level_max
            <scalar_t>
                <<<grid_dim_l, block_dim>>>
                    (access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        access.get_y_vec_ptr(),
                        tree_access.get_level_xz_vec_ptr(),
                        tree_access.get_xz_end_vec_ptr(),
                        tree_access.get_y_vec_ptr(),
                        input,
                        output,
                        max_indices,
                        access.z_num(current_max_level),
                        access.x_num(current_max_level),
                        tree_access.x_num(current_max_level - 1),
                        current_max_level,
                        batch,
                        tree_offset_out,
                        ne_rows_gpu.get() + offset);
    }
    cudaDeviceSynchronize();
}


template<typename scalar_t>
void apply_maxpool_backward(torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                            const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                            const torch::PackedTensorAccessor32<int64_t, 3> max_indices,
                            const size_t num_fixed_particles,
                            const size_t num_tree_particles,
                            const int batch) {

    const int num_channels = grad_input.size(1);

    // copy gradients for particles up to current_level_max
    int block_dim, grid_dim, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_dim, copy_kernel<scalar_t>, 0, 0);
    grid_dim = std::min((num_fixed_particles + block_dim - 1) / block_dim, (size_t) min_grid_size);

    if(grid_dim > 0) {
        copy_kernel<scalar_t>
            <<<grid_dim, block_dim>>>(grad_output, grad_input, num_fixed_particles, batch);
    }

    grid_dim = std::min((num_tree_particles + block_dim - 1) / block_dim, (size_t) min_grid_size);

    if(grid_dim > 0) {
        unpool_gradient<scalar_t>
            <<<grid_dim, block_dim>>>(grad_input, grad_output, max_indices, num_fixed_particles, num_tree_particles, batch);
    }
}


template __global__ void copy_kernel<float>(
        const torch::PackedTensorAccessor32<float, 3>,
        torch::PackedTensorAccessor32<float, 3>,
        const size_t, const int);

template __global__ void copy_kernel<double>(
        const torch::PackedTensorAccessor32<double, 3>,
        torch::PackedTensorAccessor32<double, 3>,
        const size_t, const int);


template void apply_maxpool<float>(
        GPUAccessHelper&, GPUAccessHelper&, 
        const torch::PackedTensorAccessor32<float, 3>,
        torch::PackedTensorAccessor32<float, 3>, 
        torch::PackedTensorAccessor32<int64_t, 3>, 
        const int, const int);

template void apply_maxpool<double>(
        GPUAccessHelper&, GPUAccessHelper&, 
        const torch::PackedTensorAccessor32<double, 3>,
        torch::PackedTensorAccessor32<double, 3>, 
        torch::PackedTensorAccessor32<int64_t, 3>, 
        const int, const int);


template void apply_maxpool_backward<float>(
        torch::PackedTensorAccessor32<float, 3>, 
        const torch::PackedTensorAccessor32<float, 3>,
        const torch::PackedTensorAccessor32<int64_t, 3>, 
        const size_t, const size_t, const int);

template void apply_maxpool_backward<double>(
        torch::PackedTensorAccessor32<double, 3>, 
        const torch::PackedTensorAccessor32<double, 3>,
        const torch::PackedTensorAccessor32<int64_t, 3>, 
        const size_t, const size_t, const int);
