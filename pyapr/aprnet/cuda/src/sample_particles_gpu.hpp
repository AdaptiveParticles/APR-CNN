//
// Created by joel on 08.03.2024.
//

#ifndef APRNET_SAMPLE_PARTICLES_GPU_HPP
#define APRNET_SAMPLE_PARTICLES_GPU_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <numerics/miscCuda.hpp>

template<typename scalar_t>
void apply_sample_particles(GPUAccessHelper &access,
							GPUAccessHelper &tree_access,
	                        const torch::PackedTensorAccessor32<scalar_t, 5> input,
	                        torch::PackedTensorAccessor32<scalar_t, 3> output,
    	                    const int min_occupied_level,
        	                const int batch,
            	            const int level_delta);


template<typename scalar_t>
void apply_sample_particles_backward(GPUAccessHelper &access,
                                	 GPUAccessHelper &tree_access,
                                	 torch::PackedTensorAccessor32<scalar_t, 5> grad_input,
                                	 const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                                	 const int min_occupied_level,
                                	 const int batch,
                                	 const int level_delta);

#endif // APRNET_SAMPLE_PARTICLES_GPU_HPP
