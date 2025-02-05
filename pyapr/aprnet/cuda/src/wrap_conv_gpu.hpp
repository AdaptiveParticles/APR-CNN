//
// Created by joel on 06.05.22.
//

#ifndef APRNET_WRAP_CONV_GPU_HPP
#define APRNET_WRAP_CONV_GPU_HPP

#include <vector>
#include "common.hpp"
#include "conv_gpu.hpp"
#include "restrict_gpu.hpp"
#include <data_structures/APR/APR.hpp>


template<typename scalar_t>
void run_conv333(std::vector<APR*> &aprs,
                 const torch::PackedTensorAccessor32<scalar_t, 3> input,
                 const torch::PackedTensorAccessor32<scalar_t, 6> weights,
                 torch::PackedTensorAccessor32<scalar_t, 2> tree_data,
                 torch::PackedTensorAccessor32<scalar_t, 3> output,
                 torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        apply_conv333<scalar_t>(access, tree_access, input, weights, tree_data, output, batch, level_delta[batch]);
    }
}


torch::Tensor conv333_forward(
        std::vector<APR*>& aprs,
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor level_delta
    ) {

    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    size_t max_tree_parts = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        auto it = (*aprs[i]).tree_iterator();
        auto dlvl = level_delta[i].item<int>();
        CHECK_LEVEL_DELTA(dlvl, it.level_max()+1);
        
        max_tree_parts = std::max(max_tree_parts, it.total_number_particles(it.level_max() - dlvl));
    }

    auto output = torch::zeros({input.size(0), weights.size(1), input.size(2)}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    auto tree_data = torch::zeros({input.size(1), (long)max_tree_parts}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "conv333_dispatch", ([&] {
        run_conv333<scalar_t>(
                aprs,
                input.packed_accessor32<scalar_t, 3>(),
                weights.packed_accessor32<scalar_t, 6>(),
                tree_data.packed_accessor32<scalar_t, 2>(),
                output.packed_accessor32<scalar_t, 3>(),
                level_delta.accessor<int, 1>()
            );
    }));

    return output;
}


template<typename scalar_t>
void run_conv333_backward(std::vector<APR*> &aprs,
                          const torch::PackedTensorAccessor32<scalar_t, 3> input,
                          const torch::PackedTensorAccessor32<scalar_t, 6> weights,
                          torch::PackedTensorAccessor32<scalar_t, 2> tree_data,
                          const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                          torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                          torch::PackedTensorAccessor32<scalar_t, 6> grad_weights,
                          torch::PackedTensorAccessor32<scalar_t, 2> grad_tree,
                          torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        apply_conv333_backward<scalar_t>(access, tree_access, input, weights, tree_data, grad_output, grad_input, grad_weights, grad_tree, batch, level_delta[batch]);
    }
}


std::vector<torch::Tensor> conv333_backward(
        std::vector<APR*>& aprs,
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor grad_output,
        torch::Tensor level_delta
    ) {

    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_output);

    size_t max_tree_parts = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        auto it = (*aprs[i]).tree_iterator();
        auto dlvl = level_delta[i].item<int>();
        CHECK_LEVEL_DELTA(dlvl, it.level_max()+1);

        max_tree_parts = std::max(max_tree_parts, it.total_number_particles(it.level_max() - dlvl));
    }

    auto tree_data = torch::zeros({input.size(1), (long)max_tree_parts}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_tree = torch::zeros_like(tree_data);

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv333_backward_dispatch", ([&] {
        run_conv333_backward<scalar_t>(
                aprs,
                input.packed_accessor32<scalar_t, 3>(),
                weights.packed_accessor32<scalar_t, 6>(),
                tree_data.packed_accessor32<scalar_t, 2>(),
                grad_output.packed_accessor32<scalar_t, 3>(),
                grad_input.packed_accessor32<scalar_t, 3>(),
                grad_weights.packed_accessor32<scalar_t, 6>(),
                grad_tree.packed_accessor32<scalar_t, 2>(),
                level_delta.accessor<int, 1>()
            );
    }));

    return {grad_input, grad_weights};
}




template<typename scalar_t>
void run_conv111(std::vector<APR*> &aprs,
                 const torch::PackedTensorAccessor32<scalar_t, 3> input,
                 const torch::PackedTensorAccessor32<scalar_t, 6> weights,
                 torch::PackedTensorAccessor32<scalar_t, 3> output,
                 torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        apply_conv111<scalar_t>(access, tree_access, input, weights, output, batch, level_delta[batch]);
    }

    torch::cuda::synchronize();
}


torch::Tensor conv111_forward(
        std::vector<APR*>& aprs,
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor level_delta
    ) {

    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    auto output = torch::zeros({input.size(0), weights.size(1), input.size(2)}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "conv111_dispatch", ([&] {
        run_conv111<scalar_t>(
                aprs,
                input.packed_accessor32<scalar_t, 3>(),
                weights.packed_accessor32<scalar_t, 6>(),
                output.packed_accessor32<scalar_t, 3>(),
                level_delta.accessor<int, 1>()
            );
    }));

    return output;
}


template<typename scalar_t>
void run_conv111_backward(std::vector<APR*> &aprs,
                          const torch::PackedTensorAccessor32<scalar_t, 3> input,
                          const torch::PackedTensorAccessor32<scalar_t, 6> weights,
                          const torch::PackedTensorAccessor32<scalar_t, 6> weights_transposed,
                          const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                          torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                          torch::PackedTensorAccessor32<scalar_t, 6> grad_weights,
                          torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        apply_conv111_backward<scalar_t>(access, tree_access, input, weights, weights_transposed, grad_output, grad_input, grad_weights, batch, level_delta[batch]);
    }
}


std::vector<torch::Tensor> conv111_backward(
        std::vector<APR*>& aprs,
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor grad_output,
        torch::Tensor level_delta
    ) {

    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_output);

    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    auto weights_transposed = weights.clone().permute({0, 2, 1, 3, 4, 5}).contiguous(); // swap channel dimensions

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv111_backward_dispatch", ([&] {
        run_conv111_backward<scalar_t>(
                aprs,
                input.packed_accessor32<scalar_t, 3>(),
                weights.packed_accessor32<scalar_t, 6>(),
                weights_transposed.packed_accessor32<scalar_t, 6>(),
                grad_output.packed_accessor32<scalar_t, 3>(),
                grad_input.packed_accessor32<scalar_t, 3>(),
                grad_weights.packed_accessor32<scalar_t, 6>(),
                level_delta.accessor<int, 1>()
            );
    }));

    return {grad_input, grad_weights};
}


#endif //APRNET_WRAP_CONV_GPU_HPP
