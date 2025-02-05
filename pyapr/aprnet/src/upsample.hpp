//
// Created by joel on 20.10.21.
//
#ifndef APRNET_UPSAMPLE_HPP
#define APRNET_UPSAMPLE_HPP

#include <vector>
#include <typeinfo>
#include <torch/all.h>
#include "helpers.hpp"
#include <data_structures/APR/APR.hpp>

class UpSampleConst {
public:
    static torch::Tensor forward(std::vector<APR*>& aprs,
                                              torch::Tensor input,
                                              torch::Tensor level_delta);

    static torch::Tensor backward(std::vector<APR*>& aprs,
                                  torch::Tensor grad_output,
                                  torch::Tensor level_delta);


private:
    template<typename scalar_t>
    static void forward_impl(std::vector<APR*>& aprs,
                             torch::TensorAccessor<scalar_t, 3> input,
                             torch::TensorAccessor<scalar_t, 3> output,
                             torch::TensorAccessor<int, 1> level_delta);

    template<typename scalar_t>
    static void backward_impl(std::vector<APR*>& aprs,
                              torch::TensorAccessor<scalar_t, 3> grad_input,
                              torch::TensorAccessor<scalar_t, 3> grad_output,
                              torch::TensorAccessor<int, 1> level_delta);
};


torch::Tensor UpSampleConst::forward(std::vector<APR*>& aprs,
                                     torch::Tensor input,
                                     torch::Tensor level_delta) {
    size_t max_size = 0;
    int min_level_delta = level_delta[0].item<int>();
    for(size_t i = 0; i < aprs.size(); ++i) {
        min_level_delta = std::min(min_level_delta, level_delta[0].item<int>());
        max_size = std::max(max_size, helpers::number_parts_after_upsampling(*aprs[i], level_delta[i].item<int>()));
    }

    if(min_level_delta <= 0) {
        std::string err_msg = "UpSampleConst is only implemented for downsampled particles. It is currently"
                              "not possible to go beyond the original resolution. That is, input must have "
                              "`level_delta > 1`.";
        throw std::invalid_argument(err_msg);
    }

    auto output = torch::zeros({input.size(0), input.size(1), (long)max_size}, input.dtype());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "UpSampleConst::forward dispatch", ([&] {
        UpSampleConst::forward_impl<scalar_t>(
                aprs,
                input.accessor<scalar_t, 3>(),
                output.accessor<scalar_t, 3>(),
                level_delta.accessor<int, 1>());
    }));

    return output;
}


template<typename scalar_t>
void UpSampleConst::forward_impl(std::vector<APR*>& aprs,
                                 torch::TensorAccessor<scalar_t, 3> input,
                                 torch::TensorAccessor<scalar_t, 3> output,
                                 torch::TensorAccessor<int, 1> level_delta) {

    const int batch_size = aprs.size();
    const int number_channels = input.size(1);

    for(int batch = 0; batch < batch_size; ++batch) {
        auto &apr = *aprs[batch];
        const int input_max_level = apr.level_max() - level_delta[batch];
        const int output_max_level = input_max_level + 1;

        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();
        auto parent_it = apr.tree_iterator();

        const auto num_to_copy = apr_it.particles_level_end(input_max_level);
        const auto in_offset = helpers::compute_tree_offset(apr_it, tree_it, input_max_level);
        const auto out_offset = helpers::compute_tree_offset(apr_it, tree_it, output_max_level);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared) firstprivate(apr_it, tree_it, parent_it)
#endif
        for(int ch = 0; ch < number_channels; ++ch) {

            // copy APR particles up to and including current_max_level of the input
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for
#endif
            for (size_t idx = 0; idx < num_to_copy; ++idx) {
                output[batch][ch][idx] = input[batch][ch][idx];
            }

            // upsample from tree to APR particles
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic)
#endif
            for(int z = 0; z < apr_it.z_num(output_max_level); ++z) {
                for(int x = 0; x < apr_it.x_num(output_max_level); ++x) {

                    parent_it.begin(input_max_level, z/2, x/2);
                    for(apr_it.begin(output_max_level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        while(parent_it.y() < (apr_it.y() / 2)) { parent_it++; }

                        output[batch][ch][apr_it] = input[batch][ch][parent_it + in_offset];
                    }
                }
            }

            // upsample from tree to tree particles (level_delta > 1)
            if(output_max_level <= tree_it.level_max()) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic)
#endif
                for (int z = 0; z < tree_it.z_num(output_max_level); ++z) {
                    for (int x = 0; x < tree_it.x_num(output_max_level); ++x) {
                        parent_it.begin(input_max_level, z / 2, x / 2);
                        for (tree_it.begin(output_max_level, z, x); tree_it < tree_it.end(); ++tree_it) {

                            while (parent_it.y() < (tree_it.y() / 2)) { parent_it++; }

                            output[batch][ch][tree_it + out_offset] = input[batch][ch][parent_it + in_offset];
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor UpSampleConst::backward(std::vector<APR*>& aprs,
                                      torch::Tensor grad_output,
                                      torch::Tensor level_delta) {
    size_t max_size = 0;
    for(size_t i = 0; i < aprs.size(); ++i) {
        max_size = std::max(max_size, helpers::number_parts(*aprs[i], level_delta[i].item<int>()));
    }

    auto grad_input = torch::zeros({grad_output.size(0), grad_output.size(1), (long)max_size}, grad_output.dtype());

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "UpSampleConst::backward dispatch", ([&] {
        UpSampleConst::backward_impl<scalar_t>(
                aprs,
                grad_input.accessor<scalar_t, 3>(),
                grad_output.accessor<scalar_t, 3>(),
                level_delta.accessor<int, 1>());
    }));

    return grad_input;
}


template<typename scalar_t>
void UpSampleConst::backward_impl(std::vector<APR*>& aprs,
                                  torch::TensorAccessor<scalar_t, 3> grad_input,
                                  torch::TensorAccessor<scalar_t, 3> grad_output,
                                  torch::TensorAccessor<int, 1> level_delta) {

    const int batch_size = aprs.size();
    const int number_channels = grad_input.size(1);

    for(int batch = 0; batch < batch_size; ++batch) {
        auto &apr = *aprs[batch];
        const int input_max_level = apr.level_max() - level_delta[batch];
        const int output_max_level = input_max_level + 1;

        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();
        auto parent_it = apr.tree_iterator();

        const auto num_to_copy = apr_it.particles_level_end(input_max_level);
        const auto in_offset = helpers::compute_tree_offset(apr_it, tree_it, input_max_level);
        const auto out_offset = helpers::compute_tree_offset(apr_it, tree_it, output_max_level);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared) firstprivate(apr_it, tree_it, parent_it)
#endif
        for(int ch = 0; ch < number_channels; ++ch) {

            // copy APR particles up to and including current_max_level of the input
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for
#endif
            for (size_t idx = 0; idx < num_to_copy; ++idx) {
                grad_input[batch][ch][idx] = grad_output[batch][ch][idx];
            }

            // upsample from tree to APR particles
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic, 2)
#endif
            for(int z = 0; z < apr_it.z_num(output_max_level); ++z) {
                for(int x = 0; x < apr_it.x_num(output_max_level); ++x) {

                    parent_it.begin(input_max_level, z/2, x/2);
                    for(apr_it.begin(output_max_level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        while(parent_it.y() < (apr_it.y() / 2)) { parent_it++; }

                        grad_input[batch][ch][parent_it + in_offset] += grad_output[batch][ch][apr_it];
                    }
                }
            }

            // upsample from tree to tree particles (level_delta > 1)
            if(output_max_level <= tree_it.level_max()) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic, 2)
#endif
                for (int z = 0; z < tree_it.z_num(output_max_level); ++z) {
                    for (int x = 0; x < tree_it.x_num(output_max_level); ++x) {
                        parent_it.begin(input_max_level, z / 2, x / 2);
                        for (tree_it.begin(output_max_level, z, x); tree_it < tree_it.end(); ++tree_it) {

                            while (parent_it.y() < (tree_it.y() / 2)) { parent_it++; }

                            grad_input[batch][ch][parent_it + in_offset] += grad_output[batch][ch][tree_it + out_offset];
                        }
                    }
                }
            }
        }
    }
}


#endif //APRNET_UPSAMPLE_HPP
