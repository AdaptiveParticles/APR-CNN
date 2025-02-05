//
// Created by joel on 18.10.22.
//

#ifndef APRNET_WRAP_RESTRICT_GPU_HPP
#define APRNET_WRAP_RESTRICT_GPU_HPP

#include <vector>
#include "common.hpp"
#include "restrict_gpu.hpp"


torch::Tensor restrict_kernel_333_forward(
        torch::Tensor weights, 
        const int num_levels) {
    
    CHECK_INPUT(weights);
    TORCH_CHECK(num_levels >= weights.size(0), "restrict_kernel_333_forward requires num_levels >= weights.shape[0]");
    auto output = torch::zeros({    num_levels,
                                    weights.size(1),
                                    weights.size(2),
                                    weights.size(3),
                                    weights.size(4),
                                    weights.size(5) },
                                torch::TensorOptions().dtype(weights.dtype()).device(weights.device()));
    
    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "restrict_kernel_333_forward_dispatch", ([&] {
        apply_restrict_kernel_333<scalar_t>(
                weights.packed_accessor32<scalar_t, 6>(),
                output.packed_accessor32<scalar_t, 6>()
            );
    }));

    return output;
}


torch::Tensor restrict_kernel_333_backward(
        torch::Tensor grad_weights_expanded, 
        const int num_levels) {
    
    CHECK_INPUT(grad_weights_expanded);
    TORCH_CHECK(num_levels <= grad_weights_expanded.size(0), "restrict_kernel_333_backward requires num_levels <= grad_weights_expanded.shape[0]");
    auto grad_weights = torch::zeros({  num_levels,
                                        grad_weights_expanded.size(1),
                                        grad_weights_expanded.size(2),
                                        grad_weights_expanded.size(3),
                                        grad_weights_expanded.size(4),
                                        grad_weights_expanded.size(5) },
                                    torch::TensorOptions().dtype(grad_weights_expanded.dtype()).device(grad_weights_expanded.device()));
    
    AT_DISPATCH_FLOATING_TYPES(grad_weights_expanded.scalar_type(), "restrict_kernel_333_backward_dispatch", ([&] {
        apply_restrict_kernel_333_backward<scalar_t>(
                grad_weights.packed_accessor32<scalar_t, 6>(),
                grad_weights_expanded.packed_accessor32<scalar_t, 6>()
            );
    }));

    return grad_weights;
}


#endif //APRNET_WRAP_RESTRICT_GPU_HPP