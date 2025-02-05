//
// Created by joel on 19.10.21.
//

#ifndef APRNET_MAXPOOL_HPP
#define APRNET_MAXPOOL_HPP

#include <vector>
#include <typeinfo>
#include <torch/all.h>
#include "helpers.hpp"
#include <data_structures/APR/APR.hpp>

class MaxPool {
public:
    static std::vector<torch::Tensor> forward(std::vector<APR*>& aprs,
                                              torch::Tensor input,
                                              torch::Tensor level_delta);

    static torch::Tensor backward(std::vector<APR*>& aprs,
                                  torch::Tensor grad_output,
                                  torch::Tensor max_indices,
                                  torch::Tensor level_delta);


private:
    template<typename scalar_t>
    static void forward_impl(std::vector<APR*>& aprs,
                             torch::TensorAccessor<scalar_t, 3> input,
                             torch::TensorAccessor<scalar_t, 3> output,
                             torch::TensorAccessor<int64_t, 3> max_indices,
                             torch::TensorAccessor<int, 1> level_delta);

    template<typename scalar_t>
    static void backward_impl(std::vector<APR*>& aprs,
                              torch::TensorAccessor<scalar_t, 3> grad_input,
                              torch::TensorAccessor<scalar_t, 3> grad_output,
                              torch::TensorAccessor<int64_t, 3> max_indices,
                              torch::TensorAccessor<int, 1> level_delta);
};


std::vector<torch::Tensor> MaxPool::forward(std::vector<APR*>& aprs,
                                            torch::Tensor input,
                                            torch::Tensor level_delta) {
    size_t max_size = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        max_size = std::max(max_size, helpers::number_parts_after_pooling(*aprs[i], level_delta[i].item<int>()));
    }

    auto output = -1e17f * torch::ones({input.size(0), input.size(1), (long)max_size}, input.dtype());
    auto max_indices = torch::zeros({output.size(0), output.size(1), output.size(2)}, torch::kLong);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "MaxPool::forward dispatch", ([&] {
        MaxPool::forward_impl<scalar_t>(
                    aprs,
                    input.accessor<scalar_t, 3>(),
                    output.accessor<scalar_t, 3>(),
                    max_indices.accessor<int64_t, 3>(),
                    level_delta.accessor<int, 1>());
    }));

    return {output, max_indices};
}


template<typename scalar_t>
void MaxPool::forward_impl(std::vector<APR*>& aprs,
                           torch::TensorAccessor<scalar_t, 3> input,
                           torch::TensorAccessor<scalar_t, 3> output,
                           torch::TensorAccessor<int64_t, 3> max_indices,
                           torch::TensorAccessor<int, 1> level_delta) {

    const int batch_size = aprs.size();
    const int number_channels = input.size(1);

    for(int batch = 0; batch < batch_size; ++batch) {
        auto &apr = *aprs[batch];
        const int current_max_level = apr.level_max() - level_delta[batch];

        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();
        auto parent_it = apr.tree_iterator();

        const auto num_to_copy = apr_it.particles_level_end(current_max_level - 1);
        const auto out_offset = helpers::compute_tree_offset(apr_it, parent_it, current_max_level - 1);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared) firstprivate(apr_it, tree_it, parent_it)
#endif
        for(int ch = 0; ch < number_channels; ++ch) {

            // copy lower level particles
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for
#endif
            for(size_t idx = 0; idx < num_to_copy; ++idx) {
                output[batch][ch][idx] = input[batch][ch][idx];
            }

            // downsample max level APR particles
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic, 2)
#endif
            for(int z = 0; z < apr_it.z_num(current_max_level); ++z) {
                for(int x = 0; x < apr_it.x_num(current_max_level); ++x) {

                    parent_it.begin(current_max_level - 1, z / 2, x / 2);

                    for(apr_it.begin(current_max_level, z, x); apr_it < apr_it.end(); ++apr_it) {
                        while (parent_it.y() < (apr_it.y() / 2)) { parent_it++; }

                        if(input[batch][ch][apr_it] > output[batch][ch][parent_it + out_offset]) {
                            output[batch][ch][parent_it + out_offset] = input[batch][ch][apr_it];
                            max_indices[batch][ch][parent_it + out_offset] = apr_it;
                        }
                    } // y
                } // x
            } // z

            // downsample upgraded tree particles, if any
            if(current_max_level < apr_it.level_max()) {
                const auto in_offset = helpers::compute_tree_offset(apr_it, tree_it, current_max_level);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic, 2)
#endif
                for(int z = 0; z < tree_it.z_num(current_max_level); ++z) {
                    for(int x = 0; x < tree_it.x_num(current_max_level); ++x) {

                        parent_it.begin(current_max_level - 1, z / 2, x / 2);

                        for(tree_it.begin(current_max_level, z, x); tree_it < tree_it.end(); ++tree_it) {
                            while (parent_it.y() < (tree_it.y() / 2)) { parent_it++; }

                            if(input[batch][ch][tree_it + in_offset] > output[batch][ch][parent_it + out_offset]) {
                                output[batch][ch][parent_it + out_offset] = input[batch][ch][tree_it + in_offset];
                                max_indices[batch][ch][parent_it + out_offset] = tree_it + in_offset;
                            }
                        } // y
                    } // x
                } // z
            } // if
        } // ch
    } // batch
} // MaxPool::forward


torch::Tensor MaxPool::backward(std::vector<APR*>& aprs,
                                torch::Tensor grad_output,
                                torch::Tensor max_indices,
                                torch::Tensor level_delta) {
    size_t max_size = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        max_size = std::max(max_size, helpers::number_parts(*aprs[i], level_delta[i].item<int>()));
    }

    auto grad_input = torch::zeros({grad_output.size(0), grad_output.size(1), (long)max_size}, grad_output.dtype());

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "MaxPool::backward dispatch", ([&] {
        MaxPool::backward_impl<scalar_t>(
                aprs,
                grad_input.accessor<scalar_t, 3>(),
                grad_output.accessor<scalar_t, 3>(),
                max_indices.accessor<int64_t, 3>(),
                level_delta.accessor<int, 1>());
    }));

    return grad_input;
}


template<typename scalar_t>
void MaxPool::backward_impl(std::vector<APR*>& aprs,
                            torch::TensorAccessor<scalar_t, 3> grad_input,
                            torch::TensorAccessor<scalar_t, 3> grad_output,
                            torch::TensorAccessor<int64_t, 3> max_indices,
                            torch::TensorAccessor<int, 1> level_delta) {

    const int batch_size = aprs.size();
    const int number_channels = grad_input.size(1);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared)
#endif
    {
#ifdef PYAPR_HAVE_OPENMP
        const auto thread_num = omp_get_thread_num();
        const auto num_threads = omp_get_num_threads();
#else
        const int thread_num = 0;
        const int num_threads = 1;
#endif
        for (int batch = 0; batch < batch_size; ++batch) {
            auto &apr = *aprs[batch];
            auto apr_it = apr.iterator();

            const int output_level_delta = level_delta[batch] + 1;
            const auto num_fixed_parts = apr_it.particles_level_end(apr_it.level_max() - output_level_delta);
            const auto total_num_parts = helpers::number_parts(apr, output_level_delta);

            // copy gradient from fixed (APR) particles
            {
                const auto chunk_size = num_fixed_parts / num_threads;
                const auto begin_ = 0 + thread_num * chunk_size;
                const auto end_ = (thread_num == num_threads - 1) ? num_fixed_parts : begin_ + chunk_size;

                for (int ch = 0; ch < number_channels; ++ch) {
                    for (size_t idx = begin_; idx < end_; ++idx) {
                        grad_input[batch][ch][idx] = grad_output[batch][ch][idx];
                    }
                }
            }

            // propagate gradient from upgraded tree particles
            {
                const auto chunk_size = (total_num_parts - num_fixed_parts) / num_threads;
                const auto begin_ = num_fixed_parts + thread_num * chunk_size;
                const auto end_ = (thread_num == num_threads - 1) ? total_num_parts : begin_ + chunk_size;

                for (int ch = 0; ch < number_channels; ++ch) {
                    for (size_t idx = begin_; idx < end_; ++idx) {
                        const auto m_idx = max_indices[batch][ch][idx];
                        grad_input[batch][ch][m_idx] = grad_output[batch][ch][idx];
                    }
                }
            }
        }
    }
}


#endif //APRNET_MAXPOOL_HPP
