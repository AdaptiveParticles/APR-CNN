//
// Created by joel on 27.09.22.
//

#ifndef APRNET_NONEMPTY_ROWS_HPP
#define APRNET_NONEMPTY_ROWS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <numerics/miscCuda.hpp>


template<int blockSize_z, int blockSize_x>
void compute_ne_rows_cuda_with_tree(GPUAccessHelper& access, 
                                    GPUAccessHelper& tree_access, 
                                    VectorData<int>& ne_count, 
                                    ScopedCudaMemHandler<int*, JUST_ALLOC>& ne_rows_gpu, 
                                    const int blockSize,
                                    const int level_delta);


#endif // APRNET_NONEMPTY_ROWS_HPP