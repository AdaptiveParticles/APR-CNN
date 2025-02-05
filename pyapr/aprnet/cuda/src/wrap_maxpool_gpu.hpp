//
// Created by Joel Jonsson on 29.06.22.
//

#ifndef APRNET_WRAP_MAXPOOL_GPU_HPP
#define APRNET_WRAP_MAXPOOL_GPU_HPP

#include "common.hpp"
#include <data_structures/APR/APR.hpp>
#include "maxpool_gpu.hpp"
#include "aprnet/src/helpers.hpp"


template<typename scalar_t>
void run_maxpool(std::vector<APR*> &aprs,
                 const torch::PackedTensorAccessor32<scalar_t, 3> input,
                 torch::PackedTensorAccessor32<scalar_t, 3> output,
                 torch::PackedTensorAccessor32<int64_t, 3> max_indices,
                 torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        apply_maxpool<scalar_t>(access, tree_access, input, output, max_indices, batch, level_delta[batch]);
    }
}


std::vector<torch::Tensor> maxpool_forward_cuda(
        std::vector<APR*>& aprs,
        torch::Tensor input,
        torch::Tensor level_delta) {

    CHECK_INPUT(input);

    int64_t max_size = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        max_size = std::max(max_size, (int64_t) helpers::number_parts_after_pooling(*aprs[i], level_delta[i].item<int>()));
    }

    auto output = torch::zeros({input.size(0), input.size(1), max_size}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    auto max_indices = torch::zeros({input.size(0), input.size(1), max_size},  torch::TensorOptions().dtype(torch::kLong).device(input.device()));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool_forward_cuda_dispatch", ([&] {
        run_maxpool<scalar_t>(
                aprs,
                input.packed_accessor32<scalar_t, 3>(),
                output.packed_accessor32<scalar_t, 3>(),
                max_indices.packed_accessor32<int64_t, 3>(),
                level_delta.accessor<int, 1>());
    }));

    return {output, max_indices};
}


template<typename scalar_t>
void run_maxpool_backward(std::vector<APR*> &aprs,
                          torch::PackedTensorAccessor32<scalar_t, 3> grad_input,
                          const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                          const torch::PackedTensorAccessor32<int64_t, 3> max_indices,
                          torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < grad_input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        const size_t num_output_particles = helpers::number_parts_after_pooling(apr, level_delta[batch]);

        auto it = apr.iterator();
        const size_t num_fixed_particles = it.particles_level_end(it.level_max() - level_delta[batch] - 1);

        apply_maxpool_backward(grad_input, grad_output, max_indices, num_fixed_particles,
                               num_output_particles - num_fixed_particles, batch);
    }
}


torch::Tensor maxpool_backward_cuda(
        std::vector<APR*>& aprs,
        torch::Tensor grad_output,
        torch::Tensor max_indices,
        torch::Tensor level_delta) {

    CHECK_INPUT(grad_output);
    CHECK_INPUT(max_indices);

    int64_t max_size = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        max_size = std::max(max_size, (int64_t) helpers::number_parts(*aprs[i], level_delta[i].item<int>()));
    }

    auto grad_input = torch::zeros({grad_output.size(0), grad_output.size(1), max_size}, torch::TensorOptions().dtype(grad_output.dtype()).device(grad_output.device()));

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "maxpool_backward_cuda_dispatch", ([&] {
        run_maxpool_backward<scalar_t>(
                aprs,
                grad_input.packed_accessor32<scalar_t, 3>(),
                grad_output.packed_accessor32<scalar_t, 3>(),
                max_indices.packed_accessor32<int64_t, 3>(),
                level_delta.accessor<int, 1>());
    }));

    return grad_input;
}


#endif //APRNET_WRAP_MAXPOOL_GPU_HPP
