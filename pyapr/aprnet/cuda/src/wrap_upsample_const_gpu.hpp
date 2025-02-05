//
// Created by Joel Jonsson on 17.10.22.
//

#ifndef APRNET_WRAP_UPSAMPLE_CONST_GPU_HPP
#define APRNET_WRAP_UPSAMPLE_CONST_GPU_HPP

#include "common.hpp"
#include <data_structures/APR/APR.hpp>
#include "upsample_constant_gpu.hpp"
#include "aprnet/src/helpers.hpp"


template<typename scalar_t>
void run_upsample_const(std::vector<APR*> &aprs,
                        const torch::PackedTensorAccessor32<scalar_t, 3> input,
                        torch::PackedTensorAccessor32<scalar_t, 3> output,
                        torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        apply_upsample_const<scalar_t>(access, tree_access, input, output, batch, level_delta[batch] - 1);
    }
}



torch::Tensor upsample_const_forward(
        std::vector<APR*>& aprs,
        torch::Tensor input,
        torch::Tensor level_delta) {

    CHECK_INPUT(input);
    TORCH_CHECK(level_delta.min().item<int>() > 0, "upsample_const_forward assumes all inputs to have level_delta > 0");

    int64_t max_size = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        max_size = std::max(max_size, (int64_t) helpers::number_parts_after_upsampling(*aprs[i], level_delta[i].item<int>()));
    }

    auto output = torch::zeros({input.size(0), input.size(1), max_size}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_const_forward_dispatch", ([&] {
        run_upsample_const<scalar_t>(
                aprs,
                input.packed_accessor32<scalar_t, 3>(),
                output.packed_accessor32<scalar_t, 3>(),
                level_delta.accessor<int, 1>());
    }));

    return output;
}



template<typename scalar_t>
void run_upsample_const_backward(std::vector<APR*> &aprs,
                                 torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                                 const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                                 torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < grad_input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        apply_upsample_const_backward<scalar_t>(access, tree_access, grad_input, grad_output, batch, level_delta[batch] - 1);
    }
}



torch::Tensor upsample_const_backward(
        std::vector<APR*>& aprs,
        torch::Tensor grad_output,
        torch::Tensor level_delta) {

    CHECK_INPUT(grad_output);
    TORCH_CHECK(level_delta.min().item<int>() > 0, "upsample_const_backward assumes all inputs to have level_delta > 0");

    int64_t max_size = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        max_size = std::max(max_size, (int64_t) helpers::number_parts(*aprs[i], level_delta[i].item<int>()));
    }

    auto grad_input = torch::zeros({grad_output.size(0), grad_output.size(1), max_size}, torch::TensorOptions().dtype(grad_output.dtype()).device(grad_output.device()));

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_const_backward_dispatch", ([&] {
        run_upsample_const_backward<scalar_t>(
                aprs,
                grad_input.packed_accessor32<scalar_t, 3>(),
                grad_output.packed_accessor32<scalar_t, 3>(),
                level_delta.accessor<int, 1>());
    }));

    return grad_input;
}


#endif //APRNET_WRAP_UPSAMPLE_CONST_GPU_HPP