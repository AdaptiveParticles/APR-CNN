//
// Created by Joel Jonsson on 23.02.23.
//

#ifndef APRNET_WRAP_SAMPLE_PARTICLES_GPU_HPP
#define APRNET_WRAP_SAMPLE_PARTICLES_GPU_HPP

#include "common.hpp"
#include <data_structures/APR/APR.hpp>
#include "sample_particles_gpu.hpp"
#include "aprnet/src/helpers.hpp"


template<typename scalar_t>
void run_sample_particles(std::vector<APR*> &aprs,
                     const torch::PackedTensorAccessor32<scalar_t, 5> input,
                     torch::PackedTensorAccessor32<scalar_t, 3> output,
                     torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        const int min_occupied_level = helpers::min_occupied_level(apr);
        apply_sample_particles(access, tree_access, input, output, min_occupied_level, batch, level_delta[batch]);
    }
}


torch::Tensor sample_particles_forward(
        std::vector<APR*>& aprs,
        torch::Tensor input,
        torch::Tensor level_delta) {

    CHECK_INPUT(input);

	size_t num_parts = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        num_parts = std::max(num_parts, helpers::number_parts(*aprs[i], level_delta[i].item<int>()));
    }

    auto output = torch::zeros({input.size(0), input.size(1), (int64_t) num_parts}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sample_particles_forward_dispatch", ([&] {
        run_sample_particles<scalar_t>(
                aprs,
                input.packed_accessor32<scalar_t, 5>(),
                output.packed_accessor32<scalar_t, 3>(),
                level_delta.accessor<int, 1>());
    }));

    return output;
}


template<typename scalar_t>
void run_sample_particles_backward(std::vector<APR*> &aprs,
                              torch::PackedTensorAccessor32<scalar_t, 5> grad_input,
                              const torch::PackedTensorAccessor32<scalar_t, 3> grad_output,
                              torch::TensorAccessor<int, 1> level_delta) {

    for(int batch = 0; batch < grad_input.size(0); ++batch) {
        auto &apr = *aprs[batch];
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        const int min_occupied_level = helpers::min_occupied_level(apr);
        apply_sample_particles_backward(access, tree_access, grad_input, grad_output, min_occupied_level, batch, level_delta[batch]);
    }
}


torch::Tensor sample_particles_backward(
        std::vector<APR*>& aprs,
        torch::Tensor grad_output,
        torch::Tensor level_delta) {

    CHECK_INPUT(grad_output);
    
	int sz = 0, sx = 0, sy = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        sz = std::max(sz, aprs[i]->z_num(aprs[i]->level_max() - level_delta[i].item<int>()));
        sx = std::max(sx, aprs[i]->x_num(aprs[i]->level_max() - level_delta[i].item<int>()));
        sy = std::max(sy, aprs[i]->y_num(aprs[i]->level_max() - level_delta[i].item<int>()));
    }

    auto grad_input = torch::zeros({grad_output.size(0), grad_output.size(1), sz, sx, sy}, torch::TensorOptions().dtype(grad_output.dtype()).device(grad_output.device()));

    AT_DISPATCH_FLOATING_TYPES(grad_input.scalar_type(), "sample_particles_backward_dispatch", ([&] {
        run_sample_particles_backward<scalar_t>(
                aprs,
                grad_input.packed_accessor32<scalar_t, 5>(),
                grad_output.packed_accessor32<scalar_t, 3>(),
                level_delta.accessor<int, 1>());
    }));

    return grad_input;
}


#endif //APRNET_WRAP_SAMPLE_PARTICLES_GPU_HPP

