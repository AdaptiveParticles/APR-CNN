#include "sample_particles_gpu.hpp"

template<typename scalar_t>
__global__ void sample_particles_kernel(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        const torch::PackedTensorAccessor32<scalar_t, 5> input,
        torch::PackedTensorAccessor32<scalar_t, 3> output,
        const int current_max_level,
        const int level_delta,
        const int batch,
        const int64_t tree_offset,
		const int y_num) {
    
    const int level = current_max_level - blockIdx.z;
    const int z = blockIdx.y;
    const int x = blockIdx.x;
    const int size_factor = std::pow(2, (int) blockIdx.z);

    const int z_l = z / size_factor;
    const int x_l = x / size_factor;

    const int x_num_l = (gridDim.x + size_factor - 1) / size_factor;
    const int z_num_l = (gridDim.y + size_factor - 1) / size_factor;

    const int x_denom = (x_l < (x_num_l - 1)) ? size_factor : gridDim.x - x_l * size_factor;
    const int z_denom = (z_l < (z_num_l - 1)) ? size_factor : gridDim.y - z_l * size_factor;
    const int xz_denom = x_denom * z_denom;

    size_t xz_start = x_l + z_l * x_num_l + level_xz_vec[level];
    size_t global_index_begin = xz_end_vec[xz_start - 1];
    size_t global_index_end = xz_end_vec[xz_start];

    if(level < current_max_level) {
		for(size_t index = global_index_begin + threadIdx.x; index < global_index_end; index += blockDim.x) {
		
		const int y_begin = y_vec[index] * size_factor;
		const int y_end = std::min(y_begin + size_factor, y_num);
		const scalar_t denom = xz_denom * (y_end - y_begin);
            
		for(int ch = 0; ch < input.size(1); ++ch) {
                scalar_t val = 0;
                for(int y = y_begin; y < y_end; ++y) {
					val += input[batch][ch][z][x][y];
                }
				atomicAdd(&output[batch][ch][index], val/denom);
            }
        }
    } else {
        for(size_t index = global_index_begin + threadIdx.x; index < global_index_end; index += blockDim.x) {
            for(int ch = 0; ch < input.size(1); ++ch) {
                const int y = y_vec[index];
                output[batch][ch][index] = input[batch][ch][z][x][y];
            }
        }

        if(level_delta > 0) {
            xz_start = x_l + z_l * x_num_l + level_xz_vec_tree[level];
            global_index_begin = xz_end_vec_tree[xz_start - 1];
            global_index_end = xz_end_vec_tree[xz_start];

            for(size_t index = global_index_begin + threadIdx.x; index < global_index_end; index += blockDim.x) {
                for(int ch = 0; ch < input.size(1); ++ch) {
                    const int y = y_vec_tree[index];
                	output[batch][ch][index + tree_offset] = input[batch][ch][z][x][y];
                }
            }
        }
    }
}


template<typename scalar_t>
__global__ void sample_particles_kernel_grad_input(
        const uint64_t* __restrict__ level_xz_vec,
        const uint64_t* __restrict__ xz_end_vec,
        const uint16_t* __restrict__ y_vec,
        const uint64_t* __restrict__ level_xz_vec_tree,
        const uint64_t* __restrict__ xz_end_vec_tree,
        const uint16_t* __restrict__ y_vec_tree,
        torch::PackedTensorAccessor32<scalar_t, 5> grad_input,
        const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
        const int current_max_level,
        const int level_delta,
        const int batch,
        const int64_t tree_offset,
		const int y_num) {
    
    const int level = current_max_level - blockIdx.z;
    const int z = blockIdx.y;
    const int x = blockIdx.x;
    const int size_factor = std::pow(2, (int) blockIdx.z);

    const int z_l = z / size_factor;
    const int x_l = x / size_factor;
    
	const int x_num_l = (gridDim.x + size_factor - 1) / size_factor;
    const int z_num_l = (gridDim.y + size_factor - 1) / size_factor;

    const int x_denom = (x_l < (x_num_l - 1)) ? size_factor : gridDim.x - x_l * size_factor;
    const int z_denom = (z_l < (z_num_l - 1)) ? size_factor : gridDim.y - z_l * size_factor;
    const int xz_denom = x_denom * z_denom;

    size_t xz_start = x_l + z_l * x_num_l + level_xz_vec[level];
    size_t global_index_begin = xz_end_vec[xz_start - 1];
    size_t global_index_end = xz_end_vec[xz_start];

    if(level < current_max_level) {
        for(size_t index = global_index_begin + threadIdx.x; index < global_index_end; index += blockDim.x) {
			const int y_begin = y_vec[index] * size_factor;
            const int y_end = std::min(y_begin + size_factor, y_num);
			const scalar_t denom = xz_denom * (y_end - y_begin); 
			for(int ch = 0; ch < grad_input.size(1); ++ch) {
                const scalar_t val = grad_output[batch][ch][index] / denom;
				for(int y = y_begin; y < y_end; ++y) {
					grad_input[batch][ch][z][x][y] = val; 
            	}
			}
        }
    } else {
        for(size_t index = global_index_begin + threadIdx.x; index < global_index_end; index += blockDim.x) {
            for(int ch = 0; ch < grad_input.size(1); ++ch) {
                const int y = y_vec[index];
                grad_input[batch][ch][z][x][y] = grad_output[batch][ch][index];
            }
        }

        if(level_delta > 0) {
            xz_start = x_l + z_l * x_num_l + level_xz_vec_tree[level];
            global_index_begin = xz_end_vec_tree[xz_start - 1];
            global_index_end = xz_end_vec_tree[xz_start];

            for(size_t index = global_index_begin + threadIdx.x; index < global_index_end; index += blockDim.x) {
                for(int ch = 0; ch < grad_input.size(1); ++ch) {
                    const int y = y_vec_tree[index];
                	grad_input[batch][ch][z][x][y] = grad_output[batch][ch][index + tree_offset];
                }
            }
        }
    }
}


template<typename scalar_t>
void apply_sample_particles(GPUAccessHelper &access,
                        	GPUAccessHelper &tree_access,
    	                    const torch::PackedTensorAccessor32<scalar_t, 5> input,
	                        torch::PackedTensorAccessor32<scalar_t, 3> output,
        	                const int min_occupied_level,
            	            const int batch,
                	        const int level_delta) {
    
    const int current_max_level = access.level_max() - level_delta;
    dim3 grid_dim(access.x_num(current_max_level), access.z_num(current_max_level), std::max(current_max_level - min_occupied_level + 1, 1));
    dim3 block_dim(32);

    const int64_t tree_offset = access.total_number_particles(current_max_level) - (int64_t) tree_access.total_number_particles(current_max_level - 1);

    sample_particles_kernel
        <scalar_t>
            <<<grid_dim, block_dim>>>
                (access.get_level_xz_vec_ptr(),
                 access.get_xz_end_vec_ptr(),
                 access.get_y_vec_ptr(),
                 tree_access.get_level_xz_vec_ptr(),
                 tree_access.get_xz_end_vec_ptr(),
                 tree_access.get_y_vec_ptr(),
                 input,
                 output,
                 current_max_level,
                 level_delta,
                 batch,
                 tree_offset,
				 access.y_num(current_max_level));
    
    cudaDeviceSynchronize();
}


template<typename scalar_t>
void apply_sample_particles_backward(GPUAccessHelper &access,
                                	 GPUAccessHelper &tree_access,
                                	 torch::PackedTensorAccessor32<scalar_t, 5> grad_input,
                                	 const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
	                                 const int min_occupied_level,
	                                 const int batch,
	                                 const int level_delta) {
    
    const int current_max_level = access.level_max() - level_delta;
    dim3 grid_dim(access.x_num(current_max_level), access.z_num(current_max_level), std::max(current_max_level - min_occupied_level + 1, 1));
    dim3 block_dim(32);

    const int64_t tree_offset = access.total_number_particles(current_max_level) - (int64_t) tree_access.total_number_particles(current_max_level - 1);

    sample_particles_kernel_grad_input
        <scalar_t>
            <<<grid_dim, block_dim>>>
                (access.get_level_xz_vec_ptr(),
                 access.get_xz_end_vec_ptr(),
                 access.get_y_vec_ptr(),
                 tree_access.get_level_xz_vec_ptr(),
                 tree_access.get_xz_end_vec_ptr(),
                 tree_access.get_y_vec_ptr(),
                 grad_input,
                 grad_output,
                 current_max_level,
                 level_delta,
                 batch,
                 tree_offset,
				 access.y_num(current_max_level));
    
    cudaDeviceSynchronize();
}


template void apply_sample_particles<float>(
    GPUAccessHelper &, GPUAccessHelper &,
    const torch::PackedTensorAccessor32<float, 5>,
    torch::PackedTensorAccessor32<float, 3>,
    const int, const int, const int);


template void apply_sample_particles<double>(
    GPUAccessHelper &, GPUAccessHelper &,
    const torch::PackedTensorAccessor32<double, 5>,
    torch::PackedTensorAccessor32<double, 3>,
    const int, const int, const int);


template void apply_sample_particles_backward<float>(
    GPUAccessHelper &, GPUAccessHelper &,
    torch::PackedTensorAccessor32<float, 5>,
    const torch::PackedTensorAccessor32<float, 3>,
    const int, const int, const int);


template void apply_sample_particles_backward<double>(
    GPUAccessHelper &, GPUAccessHelper &,
    torch::PackedTensorAccessor32<double, 5>,
    const torch::PackedTensorAccessor32<double, 3>,
    const int, const int, const int);

