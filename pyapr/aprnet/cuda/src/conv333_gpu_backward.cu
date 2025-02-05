#include "conv_gpu.hpp"
#include "fill_tree_gpu.hpp"
#include "nonempty_rows.hpp"

#define BLOCKSIZE 4
#define CHUNKSIZE 32
#define DI_MAX_CHANNELS_IN 1
#define DI_MAX_CHANNELS_OUT 32
#define DW_MAX_CHANNELS_IN 8
#define DW_MAX_CHANNELS_OUT 8


//------------------------------------------
//----      input gradient kernels      ----
//------------------------------------------


template<typename scalar_t>
__global__ void conv_max_333_grad_input(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
        const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
        const torch::PackedTensorAccessor32<scalar_t, 6> stencil,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int level,
        const int stencil_num,
        const int batch,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset_out = blockIdx.y * DI_MAX_CHANNELS_OUT;
    const int channel_offset_in  = blockIdx.z * DI_MAX_CHANNELS_IN;

    const int num_channels_out = min(DI_MAX_CHANNELS_OUT, grad_output.size(1) - channel_offset_out);
    const int num_channels_in  = min(DI_MAX_CHANNELS_IN,  grad_input.size(1) - channel_offset_in);

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const int row = threadIdx.y + threadIdx.z * BLOCKSIZE;

    __shared__ scalar_t local_stencil[DI_MAX_CHANNELS_OUT][DI_MAX_CHANNELS_IN][3][3][3];
    __shared__ scalar_t grad_in_patch[DI_MAX_CHANNELS_IN][BLOCKSIZE][BLOCKSIZE][CHUNKSIZE];
    __shared__ scalar_t grad_out_patch[DI_MAX_CHANNELS_OUT][BLOCKSIZE-2][BLOCKSIZE-2][CHUNKSIZE];

    // copy weights to shared memory
    if(threadIdx.x < 27) {
        for(int tmp = row; tmp < num_channels_in * num_channels_out; tmp += blockDim.y * blockDim.z) {
            local_stencil[tmp % num_channels_out][tmp / num_channels_out][threadIdx.x / 9][(threadIdx.x % 9) / 3][threadIdx.x % 3] = \
                stencil[stencil_num][tmp % num_channels_out + channel_offset_out][tmp / num_channels_out + channel_offset_in][threadIdx.x / 9][(threadIdx.x % 9) / 3][threadIdx.x % 3];
        }
    }

    const bool out_of_bounds = (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ;

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < BLOCKSIZE - 1) &&
                           (threadIdx.z > 0) && (threadIdx.z < BLOCKSIZE - 1);

    __shared__ size_t global_index_begin_0_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_0_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_p_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_p_s[BLOCKSIZE * BLOCKSIZE];

    if(threadIdx.x == 0 && !out_of_bounds) {
        // particle row index range
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];

        // parent row index range
        xz_start = x_index / 2 + (z_index / 2) * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }
    __syncwarp();

    int y_0, y_p;

    size_t particle_index = global_index_begin_0_s[row] + threadIdx.x;
    size_t parent_index = global_index_begin_p_s[row] + threadIdx.x / 2;

    if(!out_of_bounds && particle_index < global_index_end_0_s[row]) {
        y_0 = y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }

    if(!out_of_bounds && parent_index < global_index_end_p_s[row]) {
        y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
    } else {
        y_p = INT32_MAX;
    }

    // overlapping y chunks
    __shared__ int chunk_end[(BLOCKSIZE-2)*(BLOCKSIZE-2)];
    __shared__ int chunk_start[(BLOCKSIZE-2)*(BLOCKSIZE-2)];

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        if(!out_of_bounds && global_index_end_0_s[row] > global_index_begin_0_s[row]) {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_0 / (CHUNKSIZE - 2);
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_vec[global_index_end_0_s[row] - 1] / (CHUNKSIZE - 2) + 1; 
        } else {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = INT32_MAX;
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = 0;
        }
    }
    __syncthreads();
    
    // reduce to find the minimal y range spanning all of the required indices
    if(threadIdx.y == 1 && threadIdx.z == 1) {
        for(unsigned int s = ((BLOCKSIZE-2)*(BLOCKSIZE-2)) / 2; s > 0; s >>= 1) {
            if(threadIdx.x < s) {
                chunk_start[threadIdx.x] = min(chunk_start[threadIdx.x], chunk_start[threadIdx.x + s]);
                chunk_end[threadIdx.x] = max(chunk_end[threadIdx.x], chunk_end[threadIdx.x + s]);
            }
            __syncwarp();
        }
    }
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();

        // reset shared memory buffers
        for(int ch = 0; ch < num_channels_in; ++ch) {
            grad_in_patch[ch][threadIdx.z][threadIdx.y][threadIdx.x] = 0;
        }
        
        if(not_ghost) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][threadIdx.x] = 0;
            }
        }
        __syncwarp();

        // update particle
        while( y_0 < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            particle_index += CHUNKSIZE;
            if(particle_index < global_index_end_0_s[row]) {
                y_0 = y_vec[particle_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent particle
        while( y_p < (y_chunk * (CHUNKSIZE - 2) - 1)) {
            parent_index += CHUNKSIZE / 2;
            if(parent_index < global_index_end_p_s[row]) {
                y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
            } else {
                y_p = INT32_MAX;
            }
        }
        __syncthreads();

        // read output gradient values into shared memory
        if(not_ghost && (y_0 >= y_chunk * (CHUNKSIZE - 2)) && (y_0 < (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][y_0 % CHUNKSIZE] = grad_output[batch][channel_offset_out + ch][particle_index];
            }
        }
        __syncthreads();

        // propagate output gradients to input gradient patch
        CONV_333_SPREAD_GRAD_INPUT(grad_out_patch, grad_in_patch, threadIdx.z, threadIdx.y, threadIdx.x)

        __syncthreads();

        // write current level input gradients to global memory
        if(y_0 <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                atomicAdd(&grad_input[batch][channel_offset_in + ch_in][particle_index], grad_in_patch[ch_in][threadIdx.z][threadIdx.y][y_0 % CHUNKSIZE]);
            }
        }
        __syncwarp();

        // write parent level input gradients to global memory
        if( (y_p <= (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                atomicAdd(&grad_input[batch][channel_offset_in + ch_in][parent_index], grad_in_patch[ch_in][threadIdx.z][threadIdx.y][y_p % CHUNKSIZE]);
            }
        }
    } // end for y_chunk
}



template<typename scalar_t>
__global__ void conv_max_333_ds_grad_input(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
        const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
        const torch::PackedTensorAccessor32<scalar_t, 6> stencil,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int level,
        const int stencil_num,
        const int batch,
        const int64_t tree_offset,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset_out = blockIdx.y * DI_MAX_CHANNELS_OUT;
    const int channel_offset_in  = blockIdx.z * DI_MAX_CHANNELS_IN;

    const int num_channels_out = min(DI_MAX_CHANNELS_OUT, grad_output.size(1) - channel_offset_out);
    const int num_channels_in  = min(DI_MAX_CHANNELS_IN,  grad_input.size(1) - channel_offset_in);

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const int row = threadIdx.y + threadIdx.z * BLOCKSIZE;

    __shared__ scalar_t local_stencil[DI_MAX_CHANNELS_OUT][DI_MAX_CHANNELS_IN][3][3][3];
    __shared__ scalar_t grad_in_patch[DI_MAX_CHANNELS_IN][BLOCKSIZE][BLOCKSIZE][CHUNKSIZE];
    __shared__ scalar_t grad_out_patch[DI_MAX_CHANNELS_OUT][BLOCKSIZE-2][BLOCKSIZE-2][CHUNKSIZE];

    // copy weights to shared memory
    if(threadIdx.x < 27) {
        for(int tmp = row; tmp < num_channels_in * num_channels_out; tmp += blockDim.y * blockDim.z) {            
            local_stencil[tmp % num_channels_out][tmp / num_channels_out][threadIdx.x / 9][(threadIdx.x % 9) / 3][threadIdx.x % 3] = \
                stencil[stencil_num][tmp % num_channels_out + channel_offset_out][tmp / num_channels_out + channel_offset_in][threadIdx.x / 9][(threadIdx.x % 9) / 3][threadIdx.x % 3];
        }
    }

    const bool out_of_bounds = (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ;

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < BLOCKSIZE - 1) &&
                           (threadIdx.z > 0) && (threadIdx.z < BLOCKSIZE - 1);

    __shared__ size_t global_index_begin_0_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_0_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_t_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_t_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_p_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_p_s[BLOCKSIZE * BLOCKSIZE];

    if(threadIdx.x == 0 && !out_of_bounds) {
        // particle row index range
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];

        // tree row index range
        xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];

        // parent row index range
        xz_start = x_index / 2 + (z_index / 2) * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }
    __syncwarp();

    int y_0, y_t, y_p;

    size_t particle_index = global_index_begin_0_s[row] + threadIdx.x;
    size_t tree_index = global_index_begin_t_s[row] + threadIdx.x;
    size_t parent_index = global_index_begin_p_s[row] + threadIdx.x / 2;

    if(!out_of_bounds && particle_index < global_index_end_0_s[row]) {
        y_0 = y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }

    if(!out_of_bounds && tree_index < global_index_end_t_s[row]) {
        y_t = y_vec_tree[tree_index];
    } else {
        y_t = INT32_MAX;
    }

    if(!out_of_bounds && parent_index < global_index_end_p_s[row]) {
        y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
    } else {
        y_p = INT32_MAX;
    }

    // overlapping y chunks
    __shared__ int chunk_end[(BLOCKSIZE-2)*(BLOCKSIZE-2)];
    __shared__ int chunk_start[(BLOCKSIZE-2)*(BLOCKSIZE-2)];

    if( ((threadIdx.x == 0) && not_ghost) ) {
        if(!out_of_bounds && global_index_end_0_s[row] > global_index_begin_0_s[row]) {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_0 / (CHUNKSIZE - 2);
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_vec[global_index_end_0_s[row] - 1] / (CHUNKSIZE - 2) + 1; 
        } else {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = INT32_MAX;
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = 0;
        }

        if(!out_of_bounds && global_index_end_t_s[row] > global_index_begin_t_s[row]) {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = min(chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)], 
                                                                             y_t / (CHUNKSIZE - 2));
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = max(chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)], 
                                                                           y_vec_tree[global_index_end_t_s[row] - 1] / (CHUNKSIZE - 2) + 1); 
        }
    }
    __syncthreads();

    // reduce to find the minimal range spanning all of the required indices
    if(threadIdx.y == 1 && threadIdx.z == 1) {
        for(unsigned int s = ((BLOCKSIZE-2)*(BLOCKSIZE-2)) / 2; s > 0; s >>= 1) {
            if(threadIdx.x < s) {
                chunk_start[threadIdx.x] = min(chunk_start[threadIdx.x], chunk_start[threadIdx.x + s]);
                chunk_end[threadIdx.x] = max(chunk_end[threadIdx.x], chunk_end[threadIdx.x + s]);
            }
            __syncwarp();
        }
    }
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {
        
        __syncthreads();
        
        // reset shared memory buffers
        for(int ch = 0; ch < num_channels_in; ++ch) {
            grad_in_patch[ch][threadIdx.z][threadIdx.y][threadIdx.x] = 0;
        }
        
        if(not_ghost) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][threadIdx.x] = 0;
            }
        }
        __syncwarp();

        // update particle
        while( y_0 < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            particle_index += CHUNKSIZE;
            if(particle_index < global_index_end_0_s[row]) {
                y_0 = y_vec[particle_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update tree particle
        while( y_t < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            tree_index += CHUNKSIZE;
            if(tree_index < global_index_end_t_s[row]) {
                y_t = y_vec_tree[tree_index];
            } else {
                y_t = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent particle
        while( y_p < (y_chunk * (CHUNKSIZE - 2) - 1)) {
            parent_index += CHUNKSIZE / 2;
            if(parent_index < global_index_end_p_s[row]) {
                y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
            } else {
                y_p = INT32_MAX;
            }
        }
        __syncthreads();

        // read output gradient values into shared memory (APR particles)
        if(not_ghost && (y_0 >= y_chunk * (CHUNKSIZE - 2)) && (y_0 < (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][y_0 % CHUNKSIZE] = grad_output[batch][channel_offset_out + ch][particle_index];
            }
        }
        __syncwarp();

        // read output gradient values into shared memory (upgraded tree particles)
        if(not_ghost && (y_t >= y_chunk * (CHUNKSIZE - 2)) && (y_t < (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][y_t % CHUNKSIZE] = grad_output[batch][channel_offset_out + ch][tree_index + tree_offset];
            }
        }
        __syncthreads();

        // propagate output gradients to input gradient patch
        CONV_333_SPREAD_GRAD_INPUT(grad_out_patch, grad_in_patch, threadIdx.z, threadIdx.y, threadIdx.x)
        __syncthreads();

        // write max level input gradients to global memory (APR particles)
        if(y_0 <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                atomicAdd(&grad_input[batch][channel_offset_in + ch_in][particle_index], grad_in_patch[ch_in][threadIdx.z][threadIdx.y][y_0 % CHUNKSIZE]);
            }
        }
        __syncwarp();

        // write max level input gradients to global memory (upgraded tree particles)
        if(y_t <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                atomicAdd(&grad_input[batch][channel_offset_in + ch_in][tree_index + tree_offset], grad_in_patch[ch_in][threadIdx.z][threadIdx.y][y_t % CHUNKSIZE]);
            }
        }
        __syncwarp();

        // write parent level input gradients to global memory
        if( (y_p <= (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                atomicAdd(&grad_input[batch][channel_offset_in + ch_in][parent_index], grad_in_patch[ch_in][threadIdx.z][threadIdx.y][y_p % CHUNKSIZE]);
            }
        }
    } // end for y_chunk
}


template<typename scalar_t>
__global__ void conv_interior_333_grad_input(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
        torch::PackedTensorAccessor32<scalar_t, 2> grad_tree,
        const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
        const torch::PackedTensorAccessor32<scalar_t, 6> stencil,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int level,
        const int stencil_num,
        const int batch,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset_out = blockIdx.y * DI_MAX_CHANNELS_OUT;
    const int channel_offset_in  = blockIdx.z * DI_MAX_CHANNELS_IN;

    const int num_channels_out = min(DI_MAX_CHANNELS_OUT, grad_output.size(1) - channel_offset_out);
    const int num_channels_in  = min(DI_MAX_CHANNELS_IN,  grad_input.size(1) - channel_offset_in);

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const int row = threadIdx.y + threadIdx.z * BLOCKSIZE;

    __shared__ scalar_t local_stencil[DI_MAX_CHANNELS_OUT][DI_MAX_CHANNELS_IN][3][3][3];
    __shared__ scalar_t grad_in_patch[DI_MAX_CHANNELS_IN][BLOCKSIZE][BLOCKSIZE][CHUNKSIZE];
    __shared__ scalar_t grad_out_patch[DI_MAX_CHANNELS_OUT][BLOCKSIZE-2][BLOCKSIZE-2][CHUNKSIZE];

    // copy weights to shared memory
    if(threadIdx.x < 27) {
        for(int tmp = row; tmp < num_channels_in * num_channels_out; tmp += blockDim.y * blockDim.z) {            
            local_stencil[tmp % num_channels_out][tmp / num_channels_out][threadIdx.x / 9][(threadIdx.x % 9) / 3][threadIdx.x % 3] = \
                stencil[stencil_num][tmp % num_channels_out + channel_offset_out][tmp / num_channels_out + channel_offset_in][threadIdx.x / 9][(threadIdx.x % 9) / 3][threadIdx.x % 3];
        }
    }

    const bool out_of_bounds = (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ;

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < BLOCKSIZE - 1) &&
                           (threadIdx.z > 0) && (threadIdx.z < BLOCKSIZE - 1);

    __shared__ size_t global_index_begin_0_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_0_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_t_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_t_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_p_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_p_s[BLOCKSIZE * BLOCKSIZE];

    if(threadIdx.x == 0 && !out_of_bounds) {
        // particle row index range
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];

        // tree row index range
        xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];

        // parent row index range
        xz_start = x_index / 2 + (z_index / 2) * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }
    __syncwarp();

    int y_0, y_t, y_p;

    size_t particle_index = global_index_begin_0_s[row] + threadIdx.x;
    size_t tree_index = global_index_begin_t_s[row] + threadIdx.x;
    size_t parent_index = global_index_begin_p_s[row] + threadIdx.x / 2;

    if(!out_of_bounds && particle_index < global_index_end_0_s[row]) {
        y_0 = y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }

    if(!out_of_bounds && tree_index < global_index_end_t_s[row]) {
        y_t = y_vec_tree[tree_index];
    } else {
        y_t = INT32_MAX;
    }

    if(!out_of_bounds && parent_index < global_index_end_p_s[row]) {
        y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
    } else {
        y_p = INT32_MAX;
    }

    // overlapping y chunks
    __shared__ int chunk_end[(BLOCKSIZE-2)*(BLOCKSIZE-2)];
    __shared__ int chunk_start[(BLOCKSIZE-2)*(BLOCKSIZE-2)];

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        if(!out_of_bounds && global_index_end_0_s[row] > global_index_begin_0_s[row]) {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_0 / (CHUNKSIZE - 2);
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_vec[global_index_end_0_s[row] - 1] / (CHUNKSIZE - 2) + 1;
        } else {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = INT32_MAX;
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = 0;
        }
    }
    __syncthreads();

    // reduce to find the minimal range spanning all of the required indices
    if(threadIdx.y == 1 && threadIdx.z == 1) {
        for(unsigned int s = ((BLOCKSIZE-2)*(BLOCKSIZE-2)) / 2; s > 0; s >>= 1) {
            if(threadIdx.x < s) {
                chunk_start[threadIdx.x] = min(chunk_start[threadIdx.x], chunk_start[threadIdx.x + s]);
                chunk_end[threadIdx.x] = max(chunk_end[threadIdx.x], chunk_end[threadIdx.x + s]);
            }
            __syncwarp();
        }
    }
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();

        // reset shared memory buffers
        for(int ch = 0; ch < num_channels_in; ++ch) {
            grad_in_patch[ch][threadIdx.z][threadIdx.y][threadIdx.x] = 0;
        }
        
        if(not_ghost) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][threadIdx.x] = 0;
            }
        }
        __syncwarp();

        // update particle
        while( y_0 < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            particle_index += CHUNKSIZE;
            if(particle_index < global_index_end_0_s[row]) {
                y_0 = y_vec[particle_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update tree particle
        while( y_t < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            tree_index += CHUNKSIZE;
            if(tree_index < global_index_end_t_s[row]) {
                y_t = y_vec_tree[tree_index];
            } else {
                y_t = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent particle
        while( y_p < (y_chunk * (CHUNKSIZE - 2) - 1)) {
            parent_index += CHUNKSIZE / 2;
            if(parent_index < global_index_end_p_s[row]) {
                y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
            } else {
                y_p = INT32_MAX;
            }
        }
        __syncthreads();

        // read output gradient values into shared memory
        if(not_ghost && (y_0 >= y_chunk * (CHUNKSIZE - 2)) && (y_0 < (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][y_0 % CHUNKSIZE] = grad_output[batch][channel_offset_out + ch][particle_index];
            }
        }
        __syncthreads();

        // propagate output gradients to input gradient patch
        CONV_333_SPREAD_GRAD_INPUT(grad_out_patch, grad_in_patch, threadIdx.z, threadIdx.y, threadIdx.x)
        __syncthreads();

        // write current level input gradients to global memory
        if(y_0 <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                atomicAdd(&grad_input[batch][channel_offset_in + ch_in][particle_index], grad_in_patch[ch_in][threadIdx.z][threadIdx.y][y_0 % CHUNKSIZE]);
            }
        }
        __syncwarp();

        // write parent level input gradients to global memory
        if( (y_p <= (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                atomicAdd(&grad_input[batch][channel_offset_in + ch_in][parent_index], grad_in_patch[ch_in][threadIdx.z][threadIdx.y][y_p % CHUNKSIZE]);
            }
        }
        __syncwarp();

        // write current level tree gradients to global memory
        if(y_t <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                atomicAdd(&grad_tree[channel_offset_in + ch_in][tree_index], grad_in_patch[ch_in][threadIdx.z][threadIdx.y][y_t % CHUNKSIZE]);
            }
        }
    } // end for y_chunk
}



//-------------------------------------------
//----      weight gradient kernels      ----
//-------------------------------------------


template<typename scalar_t>
__global__ void conv_max_333_grad_weights(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
        torch::PackedTensorAccessor32<scalar_t, 6> grad_weights,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int level,
        const int stencil_num,
        const int batch,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset_out = blockIdx.y * DW_MAX_CHANNELS_OUT;
    const int channel_offset_in  = blockIdx.z * DW_MAX_CHANNELS_IN;

    const int num_channels_out = min(DW_MAX_CHANNELS_OUT, grad_output.size(1) - channel_offset_out);
    const int num_channels_in  = min(DW_MAX_CHANNELS_IN,  input.size(1) - channel_offset_in);

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const int row = threadIdx.y + threadIdx.z * BLOCKSIZE;

    __shared__ scalar_t grad_w_local[3][3][3][CHUNKSIZE];
    __shared__ scalar_t local_patch[DW_MAX_CHANNELS_IN][BLOCKSIZE][BLOCKSIZE][CHUNKSIZE];
    __shared__ scalar_t grad_out_patch[DW_MAX_CHANNELS_OUT][BLOCKSIZE-2][BLOCKSIZE-2][CHUNKSIZE];

    const bool out_of_bounds = (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num);

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < BLOCKSIZE - 1) &&
                           (threadIdx.z > 0) && (threadIdx.z < BLOCKSIZE - 1);

    __shared__ size_t global_index_begin_0_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_0_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_p_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_p_s[BLOCKSIZE * BLOCKSIZE];

    if(threadIdx.x == 0 && !out_of_bounds) {
        // particle row index range
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
        
        // parent row index range
        xz_start = x_index / 2 + (z_index / 2) * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }
    __syncwarp();

    int y_0, y_p;

    size_t particle_index = global_index_begin_0_s[row] + threadIdx.x;
    size_t parent_index = global_index_begin_p_s[row] + threadIdx.x / 2;

    if(!out_of_bounds && particle_index < global_index_end_0_s[row]) {
        y_0 = y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }

    if(!out_of_bounds && parent_index < global_index_end_p_s[row]) {
        y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
    } else {
        y_p = INT32_MAX;
    }

    // overlapping y chunks
    __shared__ int chunk_end[(BLOCKSIZE-2)*(BLOCKSIZE-2)];
    __shared__ int chunk_start[(BLOCKSIZE-2)*(BLOCKSIZE-2)];

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        if(!out_of_bounds && global_index_end_0_s[row] > global_index_begin_0_s[row]) {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_0 / (CHUNKSIZE - 2);
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_vec[global_index_end_0_s[row] - 1] / (CHUNKSIZE - 2) + 1; 
        } else {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = INT32_MAX;
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = 0;
        }
    }
    __syncthreads();
    
    // reduce to find the minimal y range spanning all of the required indices
    if(threadIdx.y == 1 && threadIdx.z == 1) {
        for(unsigned int s = ((BLOCKSIZE-2)*(BLOCKSIZE-2)) / 2; s > 0; s >>= 1) {
            if(threadIdx.x < s) {
                chunk_start[threadIdx.x] = min(chunk_start[threadIdx.x], chunk_start[threadIdx.x + s]);
                chunk_end[threadIdx.x] = max(chunk_end[threadIdx.x], chunk_end[threadIdx.x + s]);
            }
            __syncwarp();
        }
    }
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();

        // reset shared memory buffers
        for(int ch = 0; ch < num_channels_in; ++ch) {
            local_patch[ch][threadIdx.z][threadIdx.y][threadIdx.x] = 0;
        }

        if(not_ghost) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][threadIdx.x] = 0;
            }
        }

        __syncwarp();

        // update particle
        while( y_0 < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            particle_index += CHUNKSIZE;
            if(particle_index < global_index_end_0_s[row]) {
                y_0 = y_vec[particle_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent particle
        while( y_p < (y_chunk * (CHUNKSIZE - 2) - 1)) {
            parent_index += CHUNKSIZE / 2;
            if(parent_index < global_index_end_p_s[row]) {
                y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
            } else {
                y_p = INT32_MAX;
            }
        }
        __syncthreads();

        // insert particles into patch
        if(y_0 <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                local_patch[ch_in][threadIdx.z][threadIdx.y][(y_0 + 1) % CHUNKSIZE] = input[batch][channel_offset_in + ch_in][particle_index];
            }
        }
        __syncwarp();

        // insert parent particles into patch
        if( (y_p <= (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                local_patch[ch_in][threadIdx.z][threadIdx.y][(y_p + 1) % CHUNKSIZE] = input[batch][channel_offset_in + ch_in][parent_index];
            }
        }
        __syncwarp();

        // read output gradient values into shared memory
        if(not_ghost && (y_0 >= y_chunk * (CHUNKSIZE - 2)) && (y_0 < (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][(y_0 + 1) % CHUNKSIZE] = grad_output[batch][channel_offset_out + ch][particle_index];
            }
        }
        __syncthreads();

        for(int w_index = threadIdx.z * BLOCKSIZE + threadIdx.y; w_index < 27; w_index += BLOCKSIZE*BLOCKSIZE) {
            CONV_333_ACCUMULATE_GRAD_WEIGHTS(grad_out_patch, local_patch, grad_w_local, grad_weights, w_index / 9, (w_index % 9) / 3, w_index % 3, stencil_num)
        }
    } // end for y_chunk
}


template<typename scalar_t>
__global__ void conv_max_333_ds_grad_weights(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
        torch::PackedTensorAccessor32<scalar_t, 6> grad_weights,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int level,
        const int stencil_num,
        const int batch,
        const int64_t tree_offset,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset_out = blockIdx.y * DW_MAX_CHANNELS_OUT;
    const int channel_offset_in  = blockIdx.z * DW_MAX_CHANNELS_IN;

    const int num_channels_out = min(DW_MAX_CHANNELS_OUT, grad_output.size(1) - channel_offset_out);
    const int num_channels_in  = min(DW_MAX_CHANNELS_IN,  input.size(1) - channel_offset_in);

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const int row = threadIdx.y + threadIdx.z * BLOCKSIZE;

    __shared__ scalar_t grad_w_local[3][3][3][CHUNKSIZE];
    __shared__ scalar_t local_patch[DW_MAX_CHANNELS_IN][BLOCKSIZE][BLOCKSIZE][CHUNKSIZE];
    __shared__ scalar_t grad_out_patch[DW_MAX_CHANNELS_OUT][BLOCKSIZE-2][BLOCKSIZE-2][CHUNKSIZE];

    const bool out_of_bounds = (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num);

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < BLOCKSIZE - 1) &&
                           (threadIdx.z > 0) && (threadIdx.z < BLOCKSIZE - 1);

    __shared__ size_t global_index_begin_0_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_0_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_t_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_t_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_p_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_p_s[BLOCKSIZE * BLOCKSIZE];

    if(threadIdx.x == 0 && !out_of_bounds) {
        // particle row index range
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];

        // tree row index range
        xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];

        // parent row index range
        xz_start = x_index / 2 + (z_index / 2) * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }
    __syncwarp();

    int y_0, y_t, y_p;

    size_t particle_index = global_index_begin_0_s[row] + threadIdx.x;
    size_t tree_index = global_index_begin_t_s[row] + threadIdx.x;
    size_t parent_index = global_index_begin_p_s[row] + threadIdx.x / 2;

    if(!out_of_bounds && particle_index < global_index_end_0_s[row]) {
        y_0 = y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }

    if(!out_of_bounds && tree_index < global_index_end_t_s[row]) {
        y_t = y_vec_tree[tree_index];
    } else {
        y_t = INT32_MAX;
    }

    if(!out_of_bounds && parent_index < global_index_end_p_s[row]) {
        y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
    } else {
        y_p = INT32_MAX;
    }
    __syncwarp();

    // overlapping y chunks
    __shared__ int chunk_end[(BLOCKSIZE-2)*(BLOCKSIZE-2)];
    __shared__ int chunk_start[(BLOCKSIZE-2)*(BLOCKSIZE-2)];

    if( ((threadIdx.x == 0) && not_ghost) ) {
        if(!out_of_bounds && global_index_end_0_s[row] > global_index_begin_0_s[row]) {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_0 / (CHUNKSIZE - 2);
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_vec[global_index_end_0_s[row] - 1] / (CHUNKSIZE - 2) + 1; 
        } else {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = INT32_MAX;
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = 0;
        }

        if(!out_of_bounds && global_index_end_t_s[row] > global_index_begin_t_s[row]) {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = min(chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)], 
                                                                             y_t / (CHUNKSIZE - 2));
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = max(chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)], 
                                                                           y_vec_tree[global_index_end_t_s[row] - 1] / (CHUNKSIZE - 2) + 1); 
        }
    }
    __syncthreads();

    // reduce to find the minimal range spanning all of the required indices
    if(threadIdx.y == 1 && threadIdx.z == 1) {
        for(unsigned int s = ((BLOCKSIZE-2)*(BLOCKSIZE-2)) / 2; s > 0; s >>= 1) {
            if(threadIdx.x < s) {
                chunk_start[threadIdx.x] = min(chunk_start[threadIdx.x], chunk_start[threadIdx.x + s]);
                chunk_end[threadIdx.x] = max(chunk_end[threadIdx.x], chunk_end[threadIdx.x + s]);
            }
            __syncwarp();
        }
    }
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {
        
        __syncthreads();

        // reset shared memory buffers
        for(int ch = 0; ch < num_channels_in; ++ch) {
            local_patch[ch][threadIdx.z][threadIdx.y][threadIdx.x] = 0;
        }

        if(not_ghost) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][threadIdx.x] = 0;
            }
        }
        __syncwarp();

        // update particle
        while( y_0 < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            particle_index += CHUNKSIZE;
            if(particle_index < global_index_end_0_s[row]) {
                y_0 = y_vec[particle_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update tree particle
        while( y_t < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            tree_index += CHUNKSIZE;
            if(tree_index < global_index_end_t_s[row]) {
                y_t = y_vec_tree[tree_index];
            } else {
                y_t = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent particle
        while( y_p < (y_chunk * (CHUNKSIZE - 2) - 1)) {
            parent_index += CHUNKSIZE / 2;
            if(parent_index < global_index_end_p_s[row]) {
                y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
            } else {
                y_p = INT32_MAX;
            }
        }
        __syncthreads();

        // insert particles into patch
        if(y_0 <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                local_patch[ch_in][threadIdx.z][threadIdx.y][(y_0 + 1) % CHUNKSIZE] = input[batch][channel_offset_in + ch_in][particle_index];
            }
        }
        __syncwarp();

        // insert tree particles into patch
        if(y_t <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                local_patch[ch_in][threadIdx.z][threadIdx.y][(y_t + 1) % CHUNKSIZE] = input[batch][channel_offset_in + ch_in][tree_index + tree_offset];
            }
        }
        __syncwarp();

        // insert parent particles into patch
        if( (y_p <= (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                local_patch[ch_in][threadIdx.z][threadIdx.y][(y_p + 1) % CHUNKSIZE] = input[batch][channel_offset_in + ch_in][parent_index];
            }
        }
        __syncwarp();

        // read output gradient values into shared memory
        if(not_ghost && (y_0 >= y_chunk * (CHUNKSIZE - 2)) && (y_0 < (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][(y_0 + 1) % CHUNKSIZE] = grad_output[batch][channel_offset_out + ch][particle_index];
            }
        }
        __syncwarp();

        // read output gradient values into shared memory
        if(not_ghost && (y_t >= y_chunk * (CHUNKSIZE - 2)) && (y_t < (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][(y_t + 1) % CHUNKSIZE] = grad_output[batch][channel_offset_out + ch][tree_index + tree_offset];
            }
        }
        __syncthreads();

        for(int w_index = threadIdx.z * BLOCKSIZE + threadIdx.y; w_index < 27; w_index += BLOCKSIZE*BLOCKSIZE) {
            CONV_333_ACCUMULATE_GRAD_WEIGHTS(grad_out_patch, local_patch, grad_w_local, grad_weights, w_index / 9, (w_index % 9) / 3, w_index % 3, stencil_num)
        }
    } // end for y_chunk
}


template<typename scalar_t>
__global__ void conv_interior_333_grad_weights(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        const torch::PackedTensorAccessor32<scalar_t, 3> input,
        const torch::PackedTensorAccessor32<scalar_t, 2> tree_data,
        const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
        torch::PackedTensorAccessor32<scalar_t, 6> grad_weights,
        const int z_num,
        const int x_num,
        const int y_num,
        const int z_num_parent,
        const int x_num_parent,
        const int level,
        const int stencil_num,
        const int batch,
        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int channel_offset_out = blockIdx.y * DW_MAX_CHANNELS_OUT;
    const int channel_offset_in  = blockIdx.z * DW_MAX_CHANNELS_IN;

    const int num_channels_out = min(DW_MAX_CHANNELS_OUT, grad_output.size(1) - channel_offset_out);
    const int num_channels_in  = min(DW_MAX_CHANNELS_IN,  input.size(1) - channel_offset_in);

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const int row = threadIdx.y + threadIdx.z * BLOCKSIZE;

    __shared__ scalar_t grad_w_local[3][3][3][CHUNKSIZE];
    __shared__ scalar_t local_patch[DW_MAX_CHANNELS_IN][BLOCKSIZE][BLOCKSIZE][CHUNKSIZE];
    __shared__ scalar_t grad_out_patch[DW_MAX_CHANNELS_OUT][BLOCKSIZE-2][BLOCKSIZE-2][CHUNKSIZE];

    const bool out_of_bounds = (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num);

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < BLOCKSIZE - 1) &&
                           (threadIdx.z > 0) && (threadIdx.z < BLOCKSIZE - 1);

    __shared__ size_t global_index_begin_0_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_0_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_t_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_t_s[BLOCKSIZE * BLOCKSIZE];

    __shared__ size_t global_index_begin_p_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ size_t global_index_end_p_s[BLOCKSIZE * BLOCKSIZE];

    if(threadIdx.x == 0 && !out_of_bounds) {
        // particle row index range
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];

        // tree row index range
        xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];

        // parent row index range
        xz_start = x_index / 2 + (z_index / 2) * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }
    __syncwarp();

    int y_0, y_t, y_p;

    size_t particle_index = global_index_begin_0_s[row] + threadIdx.x;
    size_t tree_index = global_index_begin_t_s[row] + threadIdx.x;
    size_t parent_index = global_index_begin_p_s[row] + threadIdx.x / 2;

    if(!out_of_bounds && particle_index < global_index_end_0_s[row]) {
        y_0 = y_vec[particle_index];
    } else {
        y_0 = INT32_MAX;
    }

    if(!out_of_bounds && tree_index < global_index_end_t_s[row]) {
        y_t = y_vec_tree[tree_index];
    } else {
        y_t = INT32_MAX;
    }

    if(!out_of_bounds && parent_index < global_index_end_p_s[row]) {
        y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
    } else {
        y_p = INT32_MAX;
    }

    // overlapping y chunks
    __shared__ int chunk_end[(BLOCKSIZE-2)*(BLOCKSIZE-2)];
    __shared__ int chunk_start[(BLOCKSIZE-2)*(BLOCKSIZE-2)];

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        if(!out_of_bounds && global_index_end_0_s[row] > global_index_begin_0_s[row]) {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_0 / (CHUNKSIZE - 2);
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = y_vec[global_index_end_0_s[row] - 1] / (CHUNKSIZE - 2) + 1;
        } else {
            chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = INT32_MAX;
            chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(BLOCKSIZE-2)] = 0;
        }
    }
    __syncthreads();

    // reduce to find the minimal range spanning all of the required indices
    if(threadIdx.y == 1 && threadIdx.z == 1) {
        for(unsigned int s = ((BLOCKSIZE-2)*(BLOCKSIZE-2)) / 2; s > 0; s >>= 1) {
            if(threadIdx.x < s) {
                chunk_start[threadIdx.x] = min(chunk_start[threadIdx.x], chunk_start[threadIdx.x + s]);
                chunk_end[threadIdx.x] = max(chunk_end[threadIdx.x], chunk_end[threadIdx.x + s]);
            }
            __syncwarp();
        }
    }
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();

        // reset shared memory buffers
        for(int ch = 0; ch < num_channels_in; ++ch) {
            local_patch[ch][threadIdx.z][threadIdx.y][threadIdx.x] = 0;
        }

        if(not_ghost) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][threadIdx.x] = 0;
            }
        }
        __syncwarp();

        // update particle
        while( y_0 < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            particle_index += CHUNKSIZE;
            if(particle_index < global_index_end_0_s[row]) {
                y_0 = y_vec[particle_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update tree particle
        while( y_t < (y_chunk * (CHUNKSIZE - 2) - 1) ) {
            tree_index += CHUNKSIZE;
            if(tree_index < global_index_end_t_s[row]) {
                y_t = y_vec_tree[tree_index];
            } else {
                y_t = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent particle
        while( y_p < (y_chunk * (CHUNKSIZE - 2) - 1)) {
            parent_index += CHUNKSIZE / 2;
            if(parent_index < global_index_end_p_s[row]) {
                y_p = 2 * y_vec[parent_index] + threadIdx.x % 2;
            } else {
                y_p = INT32_MAX;
            }
        }
        __syncthreads();

        // insert particles into patch
        if(y_0 <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                local_patch[ch_in][threadIdx.z][threadIdx.y][(y_0 + 1) % CHUNKSIZE] = input[batch][channel_offset_in + ch_in][particle_index];
            }
        }
        __syncwarp();

        // insert tree particles into patch
        if(y_t <= (y_chunk + 1) * (CHUNKSIZE - 2)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                local_patch[ch_in][threadIdx.z][threadIdx.y][(y_t + 1) % CHUNKSIZE] = tree_data[channel_offset_in + ch_in][tree_index];
            }
        }
        __syncwarp();

        // insert parent particles into patch
        if( (y_p <= (y_chunk + 1) * (CHUNKSIZE - 2)) && (y_p < y_num)) {
            for(int ch_in = 0; ch_in < num_channels_in; ++ch_in) {
                local_patch[ch_in][threadIdx.z][threadIdx.y][(y_p + 1) % CHUNKSIZE] = input[batch][channel_offset_in + ch_in][parent_index];
            }
        }
        __syncwarp();

        // read output gradient values into shared memory
        if(not_ghost && (y_0 >= y_chunk * (CHUNKSIZE - 2)) && (y_0 < (y_chunk + 1) * (CHUNKSIZE - 2))) {
            for(int ch = 0; ch < num_channels_out; ++ch) {
                grad_out_patch[ch][threadIdx.z-1][threadIdx.y-1][(y_0 + 1) % CHUNKSIZE] = grad_output[batch][channel_offset_out + ch][particle_index];
            }
        }
        __syncthreads();

        for(int w_index = threadIdx.z * BLOCKSIZE + threadIdx.y; w_index < 27; w_index += BLOCKSIZE*BLOCKSIZE) {
            CONV_333_ACCUMULATE_GRAD_WEIGHTS(grad_out_patch, local_patch, grad_w_local, grad_weights, w_index / 9, (w_index % 9) / 3, w_index % 3, stencil_num)
        }
    } // end for y_chunk
}


template<typename scalar_t>
void apply_conv333_backward(GPUAccessHelper &access,
                            GPUAccessHelper &tree_access,
                            const torch::PackedTensorAccessor32<scalar_t, 3> input,
                            const torch::PackedTensorAccessor32<scalar_t, 6> weights,
                            torch::PackedTensorAccessor32<scalar_t, 2> tree_data,
                            const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                            torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                            torch::PackedTensorAccessor32<scalar_t, 6> grad_weights,
                            torch::PackedTensorAccessor32<scalar_t, 2> grad_tree,
                            const int batch,
                            const int level_delta) {

    cudaMemsetAsync(grad_tree.data(), 0, grad_tree.size(0) * grad_tree.size(1) * sizeof(scalar_t));

    // compute tree data
    apply_fill_tree_mean(access, tree_access, input, tree_data, batch, level_delta);

    // non empty row locations
    VectorData<int> ne_counter;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu;
    compute_ne_rows_cuda_with_tree<16, 32>(access, tree_access, ne_counter, ne_rows_gpu, 2, level_delta);

    const int num_stencils = weights.size(0);
    const int num_out_channels = weights.size(1);
    const int num_in_channels = weights.size(2);
    const int current_max_level = access.level_max() - level_delta;

    const int num_blocks_out_di = (num_out_channels + DI_MAX_CHANNELS_OUT - 1) / DI_MAX_CHANNELS_OUT;
    const int num_blocks_in_di = (num_in_channels + DI_MAX_CHANNELS_IN - 1) / DI_MAX_CHANNELS_IN;

    const int num_blocks_out_dw = (num_out_channels + DW_MAX_CHANNELS_OUT - 1) / DW_MAX_CHANNELS_OUT;
    const int num_blocks_in_dw = (num_in_channels + DW_MAX_CHANNELS_IN - 1) / DW_MAX_CHANNELS_IN;

    int ne_sz = ne_counter[current_max_level + 1] - ne_counter[current_max_level];
    int offset = ne_counter[current_max_level];

    dim3 grid_dim_di(ne_sz, num_blocks_out_di, num_blocks_in_di);
    dim3 grid_dim_dw(ne_sz, num_blocks_out_dw, num_blocks_in_dw);
    dim3 block_dim(CHUNKSIZE, BLOCKSIZE, BLOCKSIZE);

    cudaDeviceSynchronize();

    // propagate grad_output to grad_input and grad_tree
    if(ne_sz > 0) {
        if(level_delta == 0) {
            conv_max_333_grad_input
                <scalar_t>
                    <<<grid_dim_di, block_dim>>>(
                        access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        access.get_y_vec_ptr(),
                        grad_input,
                        grad_output,
                        weights,
                        access.z_num(current_max_level),
                        access.x_num(current_max_level),
                        access.y_num(current_max_level),
                        access.z_num(current_max_level - 1),
                        access.x_num(current_max_level - 1),
                        current_max_level,
                        0,
                        batch,
                        ne_rows_gpu.get() + offset);
            
            conv_max_333_grad_weights
                <scalar_t>
                    <<<grid_dim_dw, block_dim>>>(
                        access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        access.get_y_vec_ptr(),
                        input,
                        grad_output,
                        grad_weights,
                        access.z_num(current_max_level),
                        access.x_num(current_max_level),
                        access.y_num(current_max_level),
                        access.z_num(current_max_level - 1),
                        access.x_num(current_max_level - 1),
                        current_max_level,
                        0,
                        batch,
                        ne_rows_gpu.get() + offset);
        
        } else {

            const int64_t tree_offset = access.total_number_particles(current_max_level) - (int64_t) tree_access.total_number_particles(current_max_level - 1);

            conv_max_333_ds_grad_input
                <scalar_t>
                    <<<grid_dim_di, block_dim>>>(
                        access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        access.get_y_vec_ptr(),
                        tree_access.get_level_xz_vec_ptr(),
                        tree_access.get_xz_end_vec_ptr(),
                        tree_access.get_y_vec_ptr(),
                        grad_input,
                        grad_output,
                        weights,
                        access.z_num(current_max_level),
                        access.x_num(current_max_level),
                        access.y_num(current_max_level),
                        access.z_num(current_max_level - 1),
                        access.x_num(current_max_level - 1),
                        current_max_level,
                        0,
                        batch,
                        tree_offset,
                        ne_rows_gpu.get() + offset);
            
            conv_max_333_ds_grad_weights
                <scalar_t>
                    <<<grid_dim_dw, block_dim>>>(
                        access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        access.get_y_vec_ptr(),
                        tree_access.get_level_xz_vec_ptr(),
                        tree_access.get_xz_end_vec_ptr(),
                        tree_access.get_y_vec_ptr(),
                        input,
                        grad_output,
                        grad_weights,
                        access.z_num(current_max_level),
                        access.x_num(current_max_level),
                        access.y_num(current_max_level),
                        access.z_num(current_max_level - 1),
                        access.x_num(current_max_level - 1),
                        current_max_level,
                        0,
                        batch,
                        tree_offset,
                        ne_rows_gpu.get() + offset);
        }
    }

    for(int level = current_max_level - 1; level >= access.level_min(); --level) {
        ne_sz = ne_counter[level+1] - ne_counter[level];
        offset = ne_counter[level];

        if(ne_sz == 0) {
            continue;
        }

        const int stencil_num = std::min(current_max_level - level, num_stencils - 1);

        dim3 grid_dim_l_di(ne_sz, num_blocks_out_di, num_blocks_in_di);
        dim3 grid_dim_l_dw(ne_sz, num_blocks_out_dw, num_blocks_in_dw);
        dim3 block_dim_l(CHUNKSIZE, BLOCKSIZE, BLOCKSIZE);

        conv_interior_333_grad_input
            <scalar_t>
                <<<grid_dim_l_di, block_dim_l>>>(
                        access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        access.get_y_vec_ptr(),
                        tree_access.get_level_xz_vec_ptr(),
                        tree_access.get_xz_end_vec_ptr(),
                        tree_access.get_y_vec_ptr(),
                        grad_input,
                        grad_tree,
                        grad_output,
                        weights,
                        access.z_num(level),
                        access.x_num(level),
                        access.y_num(level),
                        access.z_num(level - 1),
                        access.x_num(level - 1),
                        level,
                        stencil_num,
                        batch,
                        ne_rows_gpu.get() + offset);

        conv_interior_333_grad_weights
            <scalar_t>
                <<<grid_dim_l_dw, block_dim_l>>>(
                        access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        access.get_y_vec_ptr(),
                        tree_access.get_level_xz_vec_ptr(),
                        tree_access.get_xz_end_vec_ptr(),
                        tree_access.get_y_vec_ptr(),
                        input,
                        tree_data,
                        grad_output,
                        grad_weights,
                        access.z_num(level),
                        access.x_num(level),
                        access.y_num(level),
                        access.z_num(level - 1),
                        access.x_num(level - 1),
                        level,
                        stencil_num,
                        batch,
                        ne_rows_gpu.get() + offset);
    }

    cudaDeviceSynchronize();

    // propagate grad_tree to grad_input
    apply_fill_tree_mean_backward(access, tree_access, grad_input, grad_tree, batch, level_delta);

    cudaDeviceSynchronize();
}


template void apply_conv333_backward<float>(
    GPUAccessHelper &,
    GPUAccessHelper &,
    const torch::PackedTensorAccessor32<float, 3>,
    const torch::PackedTensorAccessor32<float, 6>,
    const torch::PackedTensorAccessor32<float, 2>,
    const torch::PackedTensorAccessor32<float, 3>,
    torch::PackedTensorAccessor32<float, 3>,
    torch::PackedTensorAccessor32<float, 6>,
    torch::PackedTensorAccessor32<float, 2>,
    const int,
    const int);


template void apply_conv333_backward<double>(
    GPUAccessHelper &,
    GPUAccessHelper &,
    const torch::PackedTensorAccessor32<double, 3>,
    const torch::PackedTensorAccessor32<double, 6>,
    const torch::PackedTensorAccessor32<double, 2>,
    const torch::PackedTensorAccessor32<double, 3>,
    torch::PackedTensorAccessor32<double, 3>,
    torch::PackedTensorAccessor32<double, 6>,
    torch::PackedTensorAccessor32<double, 2>,
    const int,
    const int);
