#include "nonempty_rows.hpp"

__forceinline__ __device__ int block_nonempty(const uint64_t* level_xz_vec,
                                              const uint64_t* xz_end_vec,
                                              const int x_num,
                                              const int level,
                                              const int x_index,
                                              const int x_limit,
                                              const int z_index,
                                              const int z_limit) {

    for(int iz = 0; iz < z_limit; ++iz) {
        for(int ix = 0; ix < x_limit; ++ix) {
            size_t xz_start = (z_index + iz) * x_num + (x_index + ix) + level_xz_vec[level];

            // if row is non-empty
            if( xz_end_vec[xz_start - 1] < xz_end_vec[xz_start]) {
                return 1;
            }
        }
    }
    return 0;
}


template<int blockSize_z, int blockSize_x>
__global__ void count_ne_rows_cuda_with_tree(const uint64_t* level_xz_vec,
                                             const uint64_t* xz_end_vec,
                                             const uint64_t* level_xz_vec_tree,
                                             const uint64_t* xz_end_vec_tree,
                                             const int z_num,
                                             const int x_num,
                                             const int level,
                                             const bool include_tree,
                                             const int chunkSize,
                                             int* res) {

    __shared__ int local_counts[blockSize_x][blockSize_z];
    local_counts[threadIdx.y][threadIdx.x] = 0;

    const int z_index = blockIdx.x * blockDim.x * chunkSize + threadIdx.x * chunkSize;

    if(z_index >= z_num) { return; } // out of bounds

    int x_index = threadIdx.y * chunkSize;

    int counter = 0;
    const int z_limit = (z_index < z_num-chunkSize) ? chunkSize : z_num-z_index;


    // loop over x-dimension in chunks
    while( x_index < x_num ) {

        const int x_limit = (x_index < x_num-chunkSize) ? chunkSize : x_num-x_index;

        int nonempty = block_nonempty(level_xz_vec, xz_end_vec, x_num, level, x_index, x_limit, z_index, z_limit);

        int nonempty_tree = 0;
        if(include_tree) {
            nonempty_tree = block_nonempty(level_xz_vec_tree, xz_end_vec_tree, x_num, level, x_index, x_limit, z_index, z_limit);
        }

        counter += (nonempty || nonempty_tree);

        x_index += blockDim.y * chunkSize;
    }
    __syncthreads();

    local_counts[threadIdx.y][threadIdx.x] = counter;
    __syncthreads();

    // reduce over blockDim.y to get the count for each z_index
    for(int gap = blockSize_x/2; gap > 0; gap/=2) {
        if(threadIdx.y < gap) {
            local_counts[threadIdx.y][threadIdx.x] += local_counts[threadIdx.y + gap][threadIdx.x];
        }
        __syncthreads();
    }

    // now reduce over blockDim.x to get the block count
    for(int gap = blockSize_z/2; gap > 0; gap/=2) {
        if(threadIdx.x < gap && threadIdx.y == 0) {
            local_counts[0][threadIdx.x] += local_counts[0][threadIdx.x + gap];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0 && threadIdx.y == 0) {
        res[blockIdx.x] = local_counts[0][0];
    }
}

__device__ unsigned int count = 0;
__global__ void fill_ne_rows_cuda_with_tree(const uint64_t* level_xz_vec,
                                            const uint64_t* xz_end_vec,
                                            const uint64_t* level_xz_vec_tree,
                                            const uint64_t* xz_end_vec_tree,
                                            const int z_num,
                                            const int x_num,
                                            const int level,
                                            const bool include_tree,
                                            const int chunkSize,
                                            unsigned int ne_count,
                                            int offset,
                                            int* ne_rows) {

    const int z_index = blockIdx.x * blockDim.x * chunkSize + threadIdx.x * chunkSize;

    if (z_index >= z_num) { return; } // out of bounds

    int x_index = threadIdx.y * chunkSize;

    const int z_limit = (z_index < z_num - chunkSize) ? chunkSize : z_num - z_index;

    // loop over x-dimension in chunks
    while (x_index < x_num) {

        const int x_limit = (x_index < x_num - chunkSize) ? chunkSize : x_num - x_index;

        // if row is non-empty
        if(block_nonempty(level_xz_vec, xz_end_vec, x_num, level, x_index, x_limit, z_index, z_limit) ||
            (include_tree && block_nonempty(level_xz_vec_tree, xz_end_vec_tree, x_num, level, x_index, x_limit, z_index, z_limit))) {
            
            unsigned int index = atomicInc(&count, ne_count-1);
            ne_rows[offset + index] = z_index * x_num + x_index;
        }

        x_index += blockDim.y * chunkSize;
    }
}


template<int blockSize_z, int blockSize_x>
void compute_ne_rows_cuda_with_tree(GPUAccessHelper& access, 
                                    GPUAccessHelper& tree_access, 
                                    VectorData<int>& ne_count, 
                                    ScopedCudaMemHandler<int*, JUST_ALLOC>& ne_rows_gpu, 
                                    const int blockSize,
                                    const int level_delta) {

    const int current_max_level = access.level_max() - level_delta;
    ne_count.resize(current_max_level + 2);

    int stride = blockSize_z * blockSize;

    int z_blocks_max = (access.z_num(current_max_level) + stride - 1) / stride;
    int num_levels = current_max_level - access.level_min() + 1;

    int block_sums_host[z_blocks_max * num_levels];
    int *block_sums_device;

    error_check(cudaMalloc(&block_sums_device, z_blocks_max*num_levels*sizeof(int)) )
    error_check( cudaMemset(block_sums_device, 0, z_blocks_max*num_levels*sizeof(int)) )

    int offset = 0;
    for(int level = access.level_min(); level <= current_max_level; ++level) {

        const bool include_tree = (level == current_max_level && level < access.level_max());

        int z_blocks = (access.z_num(level) + stride - 1) / stride;

        dim3 grid_dim(z_blocks, 1, 1);
        dim3 block_dim(blockSize_z, blockSize_x, 1);

        count_ne_rows_cuda_with_tree<blockSize_z, blockSize_x>
                <<< grid_dim, block_dim >>>(
                        access.get_level_xz_vec_ptr(),
                        access.get_xz_end_vec_ptr(),
                        tree_access.get_level_xz_vec_ptr(),
                        tree_access.get_xz_end_vec_ptr(),
                        access.z_num(level),
                        access.x_num(level),
                        level,
                        include_tree,
                        blockSize,
                        block_sums_device + offset);
        offset += z_blocks_max;
    }

    error_check(cudaDeviceSynchronize())
    error_check(cudaMemcpy(block_sums_host, block_sums_device, z_blocks_max*num_levels*sizeof(int), cudaMemcpyDeviceToHost))

    int counter = 0;
    offset = 0;

    for(int level = access.level_min(); level <= current_max_level; ++level) {
        ne_count[level] = counter;

        for(int i = 0; i < z_blocks_max; ++i) {
            counter += block_sums_host[offset + i];
        }

        offset += z_blocks_max;
    }

    ne_count.back() = counter;
    ne_rows_gpu.initialize(NULL, counter);

    for(int level = access.level_min(); level <= current_max_level; ++level) {
        
        int ne_sz = ne_count[level+1] - ne_count[level];
        if( ne_sz == 0 ) {
            continue;
        }

        const bool include_tree = (level == current_max_level && level < access.level_max());

        int z_blocks = (access.z_num(level) + blockSize_z * blockSize - 1) / (blockSize_z * blockSize);

        dim3 grid_dim(z_blocks, 1, 1);
        dim3 block_dim(blockSize_z, blockSize_x, 1);

        fill_ne_rows_cuda_with_tree
            <<< grid_dim, block_dim >>>(
                    access.get_level_xz_vec_ptr(),
                    access.get_xz_end_vec_ptr(),
                    tree_access.get_level_xz_vec_ptr(),
                    tree_access.get_xz_end_vec_ptr(),
                    access.z_num(level),
                    access.x_num(level),
                    level,
                    include_tree,
                    blockSize,
                    ne_sz,
                    ne_count[level],
                    ne_rows_gpu.get());
    }

    error_check(cudaFree(block_sums_device))
}


template void compute_ne_rows_cuda_with_tree<16, 32>(GPUAccessHelper&, GPUAccessHelper&, VectorData<int>&, ScopedCudaMemHandler<int*, JUST_ALLOC>&, const int, const int);
