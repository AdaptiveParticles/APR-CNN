//
// Created by Joel Jonsson on 29.08.22.
//

#include "fill_tree_gpu.hpp"

#define MAX_CHANNELS 8


template<typename scalar_t>
__global__ void fill_tree_mean_level_max(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 2> output,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int y_num_parent,
        const int level,
        const int batch,
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

    const bool row_within_bounds = x_index < x_num && z_index < z_num;

    // compute scale factors for averages
    scalar_t scale_factor_xz = (((2*x_num_parent != x_num) && x_p==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && z_p==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    scalar_t scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }

    if((local_th==0) ) {
        // APR particle sparse row index range
        if(row_within_bounds) {
            size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
            global_index_begin_0[block] = xz_end_vec[xz_start - 1];
            global_index_end_0[block] = xz_end_vec[xz_start];
        }
        // Parent particle sparse row index range
        size_t xz_start = x_p + z_p*x_num_parent + level_xz_vec_tree[level-1];
        global_index_begin_p[block] = xz_end_vec_tree[xz_start - 1];
        global_index_end_p[block] = xz_end_vec_tree[xz_start];
    }

    __syncwarp();

    int y_0, y_p;

    size_t particle_index = global_index_begin_0[block] + local_th;
    size_t parent_index = global_index_begin_p[block] + local_th;

    // fetch particle y index
    if (row_within_bounds && particle_index < global_index_end_0[block]) {
        y_0 =  y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }

    // fetch parent particle y index
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

        // reset local cache
        for(int ch = 0; ch < num_channels; ++ch) {
            value_cache[ch][2 * block + local_th % 2][local_th / 2] = 0;
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
        }

        __syncthreads();

        // apply downsampling
        if (y_p < (y_block+1) * 16) {
            for(int ch = block; ch < num_channels; ch += 4) {

                const scalar_t parent_value = value_cache[ch][0][y_p % 16] +
                                              value_cache[ch][1][y_p % 16] +
                                              value_cache[ch][2][y_p % 16] +
                                              value_cache[ch][3][y_p % 16] +
                                              value_cache[ch][4][y_p % 16] +
                                              value_cache[ch][5][y_p % 16] +
                                              value_cache[ch][6][y_p % 16] +
                                              value_cache[ch][7][y_p % 16];
                
                const scalar_t factor = y_p == (y_num_parent-1) ? scale_factor_yxz : scale_factor_xz;

                output[ch + channel_offset][parent_index] = factor * parent_value / scalar_t(8);
            }
        }
    }
}



template<typename scalar_t>
__global__ void fill_tree_mean_level_max_backward(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
        const torch::PackedTensorAccessor32<scalar_t, 2> grad_tree,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int y_num_parent,
        const int level,
        const int batch,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset = blockIdx.y * MAX_CHANNELS;
    const int num_channels = min(MAX_CHANNELS, grad_input.size(1) - channel_offset);

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
    __shared__ scalar_t value_cache[MAX_CHANNELS][16];

    __syncwarp();

    const bool row_within_bounds = x_index < x_num && z_index < z_num;

    // compute scale factors for averages
    scalar_t scale_factor_xz = (((2*x_num_parent != x_num) && x_p==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && z_p==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    scalar_t scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }

    if((local_th==0) ) {
        // APR particle sparse row index range
        if(row_within_bounds) {
            size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
            global_index_begin_0[block] = xz_end_vec[xz_start - 1];
            global_index_end_0[block] = xz_end_vec[xz_start];
        }
        // Parent particle sparse row index range
        size_t xz_start = x_p + z_p*x_num_parent + level_xz_vec_tree[level-1];
        global_index_begin_p[block] = xz_end_vec_tree[xz_start - 1];
        global_index_end_p[block] = xz_end_vec_tree[xz_start];
    }

    __syncwarp();

    int y_0, y_p;

    size_t particle_index = global_index_begin_0[block] + local_th;
    size_t parent_index = global_index_begin_p[block] + local_th;

    // fetch particle y index
    if (row_within_bounds && particle_index < global_index_end_0[block]) {
        y_0 =  y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }
    __syncwarp();

    // fetch parent particle y index
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
        __syncwarp();

        // update gradient cache
        if(y_p < (y_block + 1) * 16) {
            for(int ch = block; ch < num_channels; ch += 4) {
                value_cache[ch][y_p % 16] = grad_tree[ch + channel_offset][parent_index];
            }
        }

        __syncthreads();

        // propagate gradient to APR particles
        if (y_0 < (y_block+1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                const scalar_t factor = (y_0 / 2) == (y_num_parent-1) ? scale_factor_yxz : scale_factor_xz;
                grad_input[batch][ch + channel_offset][particle_index] += factor * value_cache[ch][(y_0 / 2) % 16] / scalar_t(8);
            }
        }
    }
}



template<typename scalar_t>
__global__ void fill_tree_mean_level_max_ds(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 2> output,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int y_num_parent,
        const int level,
        const int batch,
        const int64_t tree_offset_in,
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

    // compute scale factors for averages
    scalar_t scale_factor_xz = (((2*x_num_parent != x_num) && x_p==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && z_p==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    scalar_t scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }

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
        for(int ch = 0; ch < num_channels; ++ch) {
            value_cache[ch][2 * block + local_th % 2][local_th / 2] = 0;
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
        }

        if (y_t < (y_block + 1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                value_cache[ch][2 * block + y_t % 2][(y_t / 2) % 16] = input[batch][ch + channel_offset][tree_index + tree_offset_in];
            }
        }

        __syncthreads();

        // apply downsampling
        if (y_p < (y_block + 1) * 16) {
            for(int ch = block; ch < num_channels; ch += 4) {
                
                const scalar_t parent_value = value_cache[ch][0][y_p % 16] +
                                              value_cache[ch][1][y_p % 16] +
                                              value_cache[ch][2][y_p % 16] +
                                              value_cache[ch][3][y_p % 16] +
                                              value_cache[ch][4][y_p % 16] +
                                              value_cache[ch][5][y_p % 16] +
                                              value_cache[ch][6][y_p % 16] +
                                              value_cache[ch][7][y_p % 16];
                
                const scalar_t factor = y_p == (y_num_parent-1) ? scale_factor_yxz : scale_factor_xz;

                output[ch + channel_offset][parent_index] = factor * parent_value / scalar_t(8);
                
            }
        }
    }
}



template<typename scalar_t>
__global__ void fill_tree_mean_level_max_ds_backward(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
        const torch::PackedTensorAccessor32<scalar_t, 2> grad_tree,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int y_num_parent,
        const int level,
        const int batch,
        const int64_t tree_offset_in,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset = blockIdx.y * MAX_CHANNELS;
    const int num_channels = min(MAX_CHANNELS, grad_input.size(1) - channel_offset);

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
    __shared__ scalar_t value_cache[MAX_CHANNELS][16];

    // compute scale factors for averages
    scalar_t scale_factor_xz = (((2*x_num_parent != x_num) && x_p==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && z_p==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    scalar_t scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz *= 2;
    }

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
        __syncwarp();

        // update gradient cache
        if(y_p < (y_block + 1) * 16) {
            for(int ch = block; ch < num_channels; ch += 4) {
                value_cache[ch][y_p % 16] = grad_tree[ch + channel_offset][parent_index];
            }
        }
        __syncthreads();

        // propagate gradient to APR particles
        if (y_0 < (y_block+1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                const scalar_t factor = (y_0 / 2) == (y_num_parent-1) ? scale_factor_yxz : scale_factor_xz;
                grad_input[batch][ch + channel_offset][particle_index] += factor * value_cache[ch][(y_0 / 2) % 16] / scalar_t(8);
            }
        }

        // propagate gradient to upgraded tree particles
        if (y_t < (y_block+1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                const scalar_t factor = (y_t / 2) == (y_num_parent-1) ? scale_factor_yxz : scale_factor_xz;
                grad_input[batch][ch + channel_offset][tree_index + tree_offset_in] += factor * value_cache[ch][(y_t / 2) % 16] / scalar_t(8);
            }
        }
    }
}




template<typename scalar_t>
__global__ void fill_tree_mean_interior(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 2> output,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int y_num_parent,
        const int level,
        const int batch,
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

    __syncwarp();

    // compute scale factors for averages
    scalar_t scale_factor_xz = (((2*x_num_parent != x_num) && x_p==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && z_p==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    scalar_t scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }

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

        // reset local cache
        for(int ch = 0; ch < num_channels; ++ch) {
            value_cache[ch][2 * block + local_th % 2][local_th / 2] = 0;
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
        }

        if (y_t < (y_block + 1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                value_cache[ch][2 * block + y_t % 2][(y_t / 2) % 16] = output[ch + channel_offset][tree_index];
            }
        }

        __syncthreads();

        // apply downsampling
        if (y_p < (y_block + 1) * 16) {
            for(int ch = block; ch < num_channels; ch += 4) {
                
                const scalar_t parent_value = value_cache[ch][0][y_p % 16] +
                                              value_cache[ch][1][y_p % 16] +
                                              value_cache[ch][2][y_p % 16] +
                                              value_cache[ch][3][y_p % 16] +
                                              value_cache[ch][4][y_p % 16] +
                                              value_cache[ch][5][y_p % 16] +
                                              value_cache[ch][6][y_p % 16] +
                                              value_cache[ch][7][y_p % 16];
                
                const scalar_t factor = y_p == (y_num_parent-1) ? scale_factor_yxz : scale_factor_xz;

                output[ch + channel_offset][parent_index] = factor * parent_value / scalar_t(8);
                
            }
        }
    }
}



template<typename scalar_t>
__global__ void fill_tree_mean_interior_backward(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
        torch::PackedTensorAccessor32<scalar_t, 2> grad_tree,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int y_num_parent,
        const int level,
        const int batch,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset = blockIdx.y * MAX_CHANNELS;
    const int num_channels = min(MAX_CHANNELS, grad_input.size(1) - channel_offset);

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
    __shared__ scalar_t value_cache[MAX_CHANNELS][16];

    // compute scale factors for averages
    scalar_t scale_factor_xz = (((2*x_num_parent != x_num) && x_p==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && z_p==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    scalar_t scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz *= 2;
    }

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
        __syncwarp();


        // update gradient cache
        if(y_p < (y_block + 1) * 16) {
            for(int ch = block; ch < num_channels; ch += 4) {
                value_cache[ch][y_p % 16] = grad_tree[ch + channel_offset][parent_index];
            }
        }
        __syncthreads();

        // propagate gradient to APR particles
        if (y_0 < (y_block+1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                const scalar_t factor = (y_0 / 2) == (y_num_parent-1) ? scale_factor_yxz : scale_factor_xz;
                grad_input[batch][ch + channel_offset][particle_index] += factor * value_cache[ch][(y_0 / 2) % 16] / scalar_t(8);
            }
        }

        // propagate gradient to tree particles
        if (y_t < (y_block+1) * 32) {
            for(int ch = 0; ch < num_channels; ++ch) {
                const scalar_t factor = (y_t / 2) == (y_num_parent-1) ? scale_factor_yxz : scale_factor_xz;
                grad_tree[ch + channel_offset][tree_index] += factor * value_cache[ch][(y_t / 2) % 16] / scalar_t(8);
            }
        }
    }
}



template<typename scalar_t>
void apply_fill_tree_mean(
        GPUAccessHelper &access,
        GPUAccessHelper &tree_access,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        torch::PackedTensorAccessor32<scalar_t, 2> output,
        const int batch,
        const int level_delta) {

    VectorData<int> ne_counter;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu;
    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter, ne_rows_gpu);

    const int current_max_level = access.level_max() - level_delta;
    const int num_channels = input.size(1);
    const int num_channel_blocks = (num_channels + MAX_CHANNELS - 1) / MAX_CHANNELS;

    cudaDeviceSynchronize();

    for(int level = current_max_level; level >= access.level_min(); --level) {
        size_t ne_sz = ne_counter[level + 1] - ne_counter[level];
        size_t offset = ne_counter[level];

        if(ne_sz == 0) {
            continue;
        }

        dim3 grid_dim(ne_sz, num_channel_blocks);
        int block_dim = 128;

        if(level == current_max_level) {
            if(level_delta == 0) {
                fill_tree_mean_level_max
                    <scalar_t>
                        <<<grid_dim, block_dim>>>(
                            access.get_level_xz_vec_ptr(),
                            access.get_xz_end_vec_ptr(),
                            access.get_y_vec_ptr(),
                            tree_access.get_level_xz_vec_ptr(),
                            tree_access.get_xz_end_vec_ptr(),
                            tree_access.get_y_vec_ptr(),
                            input,
                            output,
                            access.z_num(level),
                            access.x_num(level),
                            access.y_num(level),
                            tree_access.z_num(level - 1),
                            tree_access.x_num(level - 1),
                            tree_access.y_num(level - 1),
                            level,
                            batch,
                            ne_rows_gpu.get() + offset);

            } else {
                const int64_t tree_offset = access.total_number_particles(level) - (int64_t) tree_access.total_number_particles(level - 1);

                fill_tree_mean_level_max_ds
                    <scalar_t>
                        <<<grid_dim, block_dim>>>(
                            access.get_level_xz_vec_ptr(),
                            access.get_xz_end_vec_ptr(),
                            access.get_y_vec_ptr(),
                            tree_access.get_level_xz_vec_ptr(),
                            tree_access.get_xz_end_vec_ptr(),
                            tree_access.get_y_vec_ptr(),
                            input,
                            output,
                            access.z_num(level),
                            access.x_num(level),
                            access.y_num(level),
                            tree_access.z_num(level - 1),
                            tree_access.x_num(level - 1),
                            tree_access.y_num(level - 1),
                            level,
                            batch,
                            tree_offset,
                            ne_rows_gpu.get() + offset);
            }
        } else {
            fill_tree_mean_interior
                <scalar_t>
                    <<<grid_dim, block_dim>>>(
                        access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        access.get_y_vec_ptr(),
                        tree_access.get_level_xz_vec_ptr(),
                        tree_access.get_xz_end_vec_ptr(),
                        tree_access.get_y_vec_ptr(),
                        input,
                        output,
                        access.z_num(level),
                        access.x_num(level),
                        access.y_num(level),
                        tree_access.z_num(level - 1),
                        tree_access.x_num(level - 1),
                        tree_access.y_num(level - 1),
                        level,
                        batch,
                        ne_rows_gpu.get() + offset);
        }
        cudaDeviceSynchronize();
    }
}



template<typename scalar_t>
void apply_fill_tree_mean_backward(
        GPUAccessHelper &access,
        GPUAccessHelper &tree_access,
        torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
        torch::PackedTensorAccessor32<scalar_t, 2> grad_tree,
        const int batch,
        const int level_delta) {

    VectorData<int> ne_counter;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu;
    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter, ne_rows_gpu);

    const int current_max_level = access.level_max() - level_delta;
    const int num_channels = grad_input.size(1);
    const int num_channel_blocks = (num_channels + MAX_CHANNELS - 1) / MAX_CHANNELS;

    cudaDeviceSynchronize();

    for(int level = access.level_min(); level <= current_max_level; ++level) {
        size_t ne_sz = ne_counter[level + 1] - ne_counter[level];
        size_t offset = ne_counter[level];

        if(ne_sz == 0) {
            continue;
        }

        dim3 grid_dim(ne_sz, num_channel_blocks);
        int block_dim = 128;

        if(level == current_max_level) {
            if(level_delta == 0) {
                fill_tree_mean_level_max_backward
                    <scalar_t>
                        <<<grid_dim, block_dim>>>(
                            access.get_level_xz_vec_ptr(),
                            access.get_xz_end_vec_ptr(),
                            access.get_y_vec_ptr(),
                            tree_access.get_level_xz_vec_ptr(),
                            tree_access.get_xz_end_vec_ptr(),
                            tree_access.get_y_vec_ptr(),
                            grad_input,
                            grad_tree,
                            access.z_num(level),
                            access.x_num(level),
                            access.y_num(level),
                            tree_access.z_num(level - 1),
                            tree_access.x_num(level - 1),
                            tree_access.y_num(level - 1),
                            level,
                            batch,
                            ne_rows_gpu.get() + offset);

            } else {
                const int64_t tree_offset = access.total_number_particles(level) - (int64_t) tree_access.total_number_particles(level - 1);

                fill_tree_mean_level_max_ds_backward
                    <scalar_t>
                        <<<grid_dim, block_dim>>>(
                            access.get_level_xz_vec_ptr(),
                            access.get_xz_end_vec_ptr(),
                            access.get_y_vec_ptr(),
                            tree_access.get_level_xz_vec_ptr(),
                            tree_access.get_xz_end_vec_ptr(),
                            tree_access.get_y_vec_ptr(),
                            grad_input,
                            grad_tree,
                            access.z_num(level),
                            access.x_num(level),
                            access.y_num(level),
                            tree_access.z_num(level - 1),
                            tree_access.x_num(level - 1),
                            tree_access.y_num(level - 1),
                            level,
                            batch,
                            tree_offset,
                            ne_rows_gpu.get() + offset);
            }
        } else {
            fill_tree_mean_interior_backward
                <scalar_t>
                    <<<grid_dim, block_dim>>>(
                        access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        access.get_y_vec_ptr(),
                        tree_access.get_level_xz_vec_ptr(),
                        tree_access.get_xz_end_vec_ptr(),
                        tree_access.get_y_vec_ptr(),
                        grad_input,
                        grad_tree,
                        access.z_num(level),
                        access.x_num(level),
                        access.y_num(level),
                        tree_access.z_num(level - 1),
                        tree_access.x_num(level - 1),
                        tree_access.y_num(level - 1),
                        level,
                        batch,
                        ne_rows_gpu.get() + offset);
        }
        cudaDeviceSynchronize();
    }
}


template void apply_fill_tree_mean<float>(
        GPUAccessHelper &, GPUAccessHelper &,
        const torch::PackedTensorAccessor32<float, 3>,
        torch::PackedTensorAccessor32<float, 2>,
        const int, const int);

template void apply_fill_tree_mean<double>(
        GPUAccessHelper &, GPUAccessHelper &,
        const torch::PackedTensorAccessor32<double, 3>,
        torch::PackedTensorAccessor32<double, 2>,
        const int, const int);



template void apply_fill_tree_mean_backward<float>(
        GPUAccessHelper &, GPUAccessHelper &,
        torch::PackedTensorAccessor32<float, 3>,
        torch::PackedTensorAccessor32<float, 2>,
        const int, const int);

template void apply_fill_tree_mean_backward<double>(
        GPUAccessHelper &, GPUAccessHelper &,
        torch::PackedTensorAccessor32<double, 3>,
        torch::PackedTensorAccessor32<double, 2>,
        const int, const int);

