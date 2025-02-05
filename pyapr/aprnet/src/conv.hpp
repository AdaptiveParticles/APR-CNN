//
// Created by joel on 05.07.21.
//

#ifndef APRNET_CONV_HPP
#define APRNET_CONV_HPP

#include <vector>
#include <string>
#include <typeinfo>
#include <torch/all.h>
#include "APRNetImageBuffer.hpp"
#include "helpers.hpp"
#include <data_structures/APR/APR.hpp>
#include <data_structures/APR/particles/ParticleData.hpp>

template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
class Convolution {
public:
    static torch::Tensor forward(std::vector<APR*>& aprs,
                                 torch::Tensor input,
                                 torch::Tensor weights,
                                 torch::Tensor level_deltas);


    static std::vector<torch::Tensor> backward(std::vector<APR*>& aprs,
                                               torch::Tensor input,
                                               torch::Tensor weights,
                                               torch::Tensor grad_output,
                                               torch::Tensor level_delta);


private:

    template<typename scalar_t>
    static void forward_impl(std::vector<APR*>& aprs,
                             torch::TensorAccessor<scalar_t, 3> input,
                             torch::TensorAccessor<scalar_t, 6> weights,
                             torch::TensorAccessor<scalar_t, 3> output,
                             torch::TensorAccessor<int, 1> level_deltas);

    template<typename scalar_t>
    static void backward_impl(std::vector<APR*>& aprs,
                              torch::TensorAccessor<scalar_t, 3> input,
                              torch::TensorAccessor<scalar_t, 6> weights,
                              torch::TensorAccessor<scalar_t, 3> grad_input,
                              torch::TensorAccessor<scalar_t, 6> grad_weights,
                              torch::TensorAccessor<scalar_t, 3> grad_output,
                              torch::TensorAccessor<int, 1> level_deltas);
};


/// forward

template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
torch::Tensor Convolution<stencilSizeZ, stencilSizeX, stencilSizeY>
                        ::forward(std::vector<APR*>& aprs,
                                  torch::Tensor input,
                                  torch::Tensor weights,
                                  torch::Tensor level_deltas) {


    if(input.size(1) != weights.size(0)) {
        std::string err_msg = "convolve: input tensor shape does not match weights tensor!";
        throw std::invalid_argument(err_msg);
    }

    auto output = torch::zeros({input.size(0), weights.size(2), input.size(2)}, input.dtype());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "Convolution::forward dispatch", ([&] {
        Convolution<stencilSizeZ, stencilSizeX, stencilSizeY>::forward_impl<scalar_t>(
                aprs,
                input.accessor<scalar_t, 3>(),
                weights.accessor<scalar_t, 6>(),
                output.accessor<scalar_t, 3>(),
                level_deltas.accessor<int, 1>());
    }));

    return output;
}


template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
template<typename scalar_t>
void Convolution<stencilSizeZ, stencilSizeX, stencilSizeY>
                ::forward_impl(std::vector<APR*>& aprs,
                               torch::TensorAccessor<scalar_t, 3> input,
                               torch::TensorAccessor<scalar_t, 6> weights,
                               torch::TensorAccessor<scalar_t, 3> output,
                               torch::TensorAccessor<int, 1> level_deltas) {

    const int batch_size = input.size(0);
    const int number_in_channels = input.size(1);
    const int num_stencils = weights.size(1);

    const int stencilHalfZ = (stencilSizeZ-1)/2;
    const int stencilHalfX = (stencilSizeX-1)/2;

    for(int batch = 0; batch < batch_size; ++batch) {

        auto& apr = *aprs[batch];
        const int current_max_level = apr.level_max() - level_deltas[batch];

        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();

        for (int ch_in = 0; ch_in < number_in_channels; ++ch_in) {

            ParticleData <scalar_t> tree_data;
            helpers::fill_tree_mean<true>(apr, input[batch][ch_in], tree_data, current_max_level);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared) firstprivate(apr_it, tree_it)
#endif
            {
                // each thread allocates a buffer for isotropic patches
                APRNetImageBuffer<scalar_t> patch_buffer(apr.org_dims(0) + stencilSizeY - 1,
                                                         stencilSizeX,
                                                         stencilSizeZ);

                std::vector<scalar_t> temp_vec(stencilSizeZ * stencilSizeX * stencilSizeY);

                for (int level = current_max_level; level >= apr_it.level_min(); --level) {
                    const int stencil_num = std::min(num_stencils - 1, current_max_level - level);
                    patch_buffer.y_num = apr_it.y_num(level) + stencilSizeY - 1;

                    const int z_num = apr_it.z_num(level);
                    const int x_num = apr_it.x_num(level);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic)
#endif
                    for (int z = 0; z < z_num; ++z) {
                        int z_start = std::max((int) z - stencilHalfZ, 0);
                        int z_end = std::min((int) z + stencilHalfZ + 1, (int) z_num);

                        // initial fill of patch_buffer
                        for (int iz = z_start; iz < z_end; ++iz) {
                            for (int ix = 0; ix <= stencilHalfX; ++ix) {
                                helpers::update_dense_array<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                        (apr_it, tree_it, level, iz, ix, input[batch][ch_in],
                                         tree_data, patch_buffer, current_max_level);
                            }
                        }

                        // zero pad boundaries
                        for (int iz = z; iz < stencilHalfZ; ++iz) {
                            helpers::zero_boundary_z<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(iz, patch_buffer);
                        }
                        for (int iz = z_num - stencilHalfZ; iz <= z; ++iz) {
                            helpers::zero_boundary_z<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(iz + 2 * stencilHalfZ, patch_buffer);
                        }
                        for (int ix = 0; ix < stencilHalfX; ++ix) {
                            helpers::zero_boundary_x<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(ix, patch_buffer);
                        }

                        helpers::accumulate_convolution<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                (apr_it, tree_it, level, z, 0, patch_buffer, weights[ch_in][stencil_num],
                                 output[batch], temp_vec, current_max_level);

                        for (int x = 1; x < x_num; ++x) {
                            if (x < x_num - stencilHalfX) {
                                z_start = std::max((int) z - stencilHalfZ, 0);
                                z_end = std::min((int) z + stencilHalfZ + 1, (int) z_num);

                                for (int iz = z_start; iz < z_end; ++iz) {
                                    helpers::update_dense_array<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                            (apr_it, tree_it, level, iz, x + stencilHalfX, input[batch][ch_in],
                                             tree_data, patch_buffer, current_max_level);
                                }

                                for (int iz = z; iz < stencilHalfZ; ++iz) {
                                    helpers::zero_boundary_z<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(
                                            iz, patch_buffer);
                                }
                                for (int iz = z_num - stencilHalfZ; iz <= z; ++iz) {
                                    helpers::zero_boundary_z<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(
                                            iz + 2 * stencilHalfZ, patch_buffer);
                                }

                            } else {
                                helpers::zero_boundary_x<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(
                                        x + 2 * stencilHalfX, patch_buffer);
                            }

                            helpers::accumulate_convolution<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                    (apr_it, tree_it, level, z, x, patch_buffer, weights[ch_in][stencil_num],
                                     output[batch], temp_vec, current_max_level);
                        } // x
                    } // z
                } // level
            } // parallel region
        } // ch_in
    } // batch
} // internal::forward


/// specialization for stencil size 1x1x1
template<>
template<typename scalar_t>
void Convolution<1, 1, 1>
        ::forward_impl(std::vector<APR*>& aprs,
                       torch::TensorAccessor<scalar_t, 3> input,
                       torch::TensorAccessor<scalar_t, 6> weights,
                       torch::TensorAccessor<scalar_t, 3> output,
                       torch::TensorAccessor<int, 1> level_deltas) {

    const int batch_size = input.size(0);
    const int number_in_channels = input.size(1);
    const int num_stencils = weights.size(1);
    const int number_out_channels = weights.size(2);

    for(int batch = 0; batch < batch_size; ++batch) {

        auto& apr = *aprs[batch];
        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();

        const int current_max_level = apr_it.level_max() - level_deltas[batch];

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for default(shared) firstprivate(apr_it, tree_it) if(number_out_channels > 15)
#endif
        for (int ch_out = 0; ch_out < number_out_channels; ++ch_out) {
            for (int ch_in = 0; ch_in < number_in_channels; ++ch_in) {
                for (int level = current_max_level; level >= apr_it.level_min(); --level) {
                    const int stencil_num = std::min(num_stencils - 1, current_max_level - level);
                    const scalar_t w = weights[ch_in][stencil_num][ch_out][0][0][0];

                    /// convolve APR particles
                    const size_t l_begin = apr_it.particles_level_begin(level);
                    const size_t l_end = apr_it.particles_level_end(level);

                    std::transform(input[batch][ch_in].data()+l_begin,
                                   input[batch][ch_in].data()+l_end,
                                   output[batch][ch_out].data()+l_begin,
                                   output[batch][ch_out].data()+l_begin,
                                   [&w](scalar_t &i, scalar_t &o){ return o + w * i; });

                    /// convolve downsampled particles
                    if (level == current_max_level && current_max_level < apr.level_max()) {
                        const auto tree_offset = helpers::compute_tree_offset(apr_it, tree_it, current_max_level);
                        const size_t l_begin_t = tree_it.particles_level_begin(current_max_level) + tree_offset;
                        const size_t l_end_t = tree_it.particles_level_end(current_max_level) + tree_offset;

                        std::transform(input[batch][ch_in].data()+l_begin_t,
                                       input[batch][ch_in].data()+l_end_t,
                                       output[batch][ch_out].data()+l_begin_t,
                                       output[batch][ch_out].data()+l_begin_t,
                                       [&w](scalar_t &i, scalar_t &o){ return o + w * i; });
                    }
                }
            }
        }
    }
}


/// backward

template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
std::vector<torch::Tensor> Convolution<stencilSizeZ, stencilSizeX, stencilSizeY>
                                    ::backward(std::vector<APR*>& aprs,
                                               torch::Tensor input,
                                               torch::Tensor weights,
                                               torch::Tensor grad_output,
                                               torch::Tensor level_deltas) {

    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "Convolution::backward dispatch", ([&] {
        Convolution<stencilSizeZ, stencilSizeX, stencilSizeY>::backward_impl<scalar_t>(
                aprs,
                input.accessor<scalar_t, 3>(),
                weights.accessor<scalar_t, 6>(),
                grad_input.accessor<scalar_t, 3>(),
                grad_weights.accessor<scalar_t, 6>(),
                grad_output.accessor<scalar_t, 3>(),
                level_deltas.accessor<int, 1>());
    }));

    return {grad_input, grad_weights};


}


template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
template<typename scalar_t>
void Convolution<stencilSizeZ, stencilSizeX, stencilSizeY>
                ::backward_impl(std::vector<APR*>& aprs,
                                torch::TensorAccessor<scalar_t, 3> input,
                                torch::TensorAccessor<scalar_t, 6> weights,
                                torch::TensorAccessor<scalar_t, 3> grad_input,
                                torch::TensorAccessor<scalar_t, 6> grad_weights,
                                torch::TensorAccessor<scalar_t, 3> grad_output,
                                torch::TensorAccessor<int, 1> level_deltas) {

    const int batch_size = input.size(0);
    const int number_in_channels = input.size(1);
    const int number_out_channels = weights.size(2);
    const int num_stencils = weights.size(1);

    const int stencilHalfZ = (stencilSizeZ-1)/2;
    const int stencilHalfX = (stencilSizeX-1)/2;

    for(int batch = 0; batch < batch_size; ++batch) {

        auto &apr = *aprs[batch];
        const int current_max_level = apr.level_max() - level_deltas[batch];

        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();

        ParticleData <scalar_t> grad_tree(tree_it.total_number_particles(current_max_level - 1));
        ParticleData <scalar_t> tree_data;

        for (int ch_in = 0; ch_in < number_in_channels; ++ch_in) {

            helpers::fill_tree_mean<true>(apr, input[batch][ch_in], tree_data, current_max_level);
            grad_tree.set_to_zero();


#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared) firstprivate(apr_it, tree_it)
#endif
            {
                // thread private buffers for input patches and gradients
                const int y_num_buf = apr_it.y_num(current_max_level) + stencilSizeY - 1;
                const size_t filter_bank_size = number_out_channels * stencilSizeZ * stencilSizeX * stencilSizeY;

                APRNetImageBuffer<scalar_t> patch_buffer(y_num_buf, stencilSizeX, stencilSizeZ);
                APRNetImageBuffer<scalar_t> grad_patch(y_num_buf, stencilSizeX, stencilSizeZ);
                std::vector<scalar_t> temp_vec(stencilSizeZ * stencilSizeX * stencilSizeY);
                std::vector<scalar_t> temp_grad_input(helpers::number_parts(apr, level_deltas[batch]), 0);
                std::vector<scalar_t> temp_dw(num_stencils*filter_bank_size, 0);
                ParticleData<scalar_t> grad_tree_local;
                grad_tree_local.data.resize(grad_tree.size(), 0);

                for (int level = current_max_level; level >= apr_it.level_min(); --level) {
                    const int stencil_num = std::min(num_stencils - 1, current_max_level - level);
                    const size_t w_offset = stencil_num * filter_bank_size;

                    // resize patch buffers
                    patch_buffer.resize(apr_it.y_num(level) + stencilSizeY - 1, stencilSizeX, stencilSizeZ, 0);
                    grad_patch.resize(apr_it.y_num(level) + stencilSizeY - 1, stencilSizeX, stencilSizeZ, 0);

                    const int z_num = apr_it.z_num(level);
                    const int x_num = apr_it.x_num(level);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic)
#endif
                    for (int z = 0; z < z_num; ++z) {

                        grad_patch.fill(0);

                        const int z_start = std::max((int) z - stencilHalfZ, 0);
                        const int z_end = std::min((int) z + stencilHalfZ + 1, (int) z_num);

                        // initial fill of patch_buffer
                        for (int iz = z_start; iz < z_end; ++iz) {
                            for (int ix = 0; ix <= stencilHalfX; ++ix) {
                                helpers::update_dense_array<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                        (apr_it, tree_it, level, iz, ix, input[batch][ch_in],
                                         tree_data, patch_buffer, current_max_level);
                            }
                        }

                        // zero pad boundaries
                        for (int iz = z; iz < stencilHalfZ; ++iz) { // lower z
                            helpers::zero_boundary_z<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(iz,
                                                                                                         patch_buffer);
                        }
                        for (int iz = z_num - stencilHalfZ; iz <= z; ++iz) { // upper z
                            helpers::zero_boundary_z<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(
                                    iz + 2 * stencilHalfZ, patch_buffer);
                        }
                        for (int ix = 0; ix < stencilHalfX; ++ix) { // lower x
                            helpers::zero_boundary_x<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(ix,
                                                                                                         patch_buffer);
                        }

                        /// dO -> grad_patch, temp_dw
                        helpers::accumulate_convolution_backward<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                (apr_it, tree_it, level, z, 0, patch_buffer, grad_patch, weights[ch_in][stencil_num],
                                 grad_output[batch], temp_dw, temp_vec, current_max_level, w_offset);

                        if (stencilHalfX > 0) {
                            helpers::zero_boundary_x<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(
                                    0, grad_patch);
                        }

                        for (int x = 1; x < x_num; ++x) {

                            if (x < x_num - stencilHalfX) {

                                // TODO: put these loops inside the functions
                                for (int iz = z_start; iz < z_end; ++iz) {
                                    helpers::update_dense_array<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                            (apr_it, tree_it, level, iz, x + stencilHalfX, input[batch][ch_in],
                                             tree_data, patch_buffer, current_max_level);
                                }
                                for (int iz = z; iz < stencilHalfZ; ++iz) {
                                    helpers::zero_boundary_z<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(
                                            iz, patch_buffer);
                                }
                                for (int iz = z_num; iz <= z + stencilHalfZ; ++iz) {
                                    helpers::zero_boundary_z<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(
                                            iz + stencilHalfZ, patch_buffer);
                                }
                            } else {
                                helpers::zero_boundary_x<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(
                                        x + 2 * stencilHalfX, patch_buffer);
                            }

                            /// dO -> grad_patch, temp_dw
                            helpers::accumulate_convolution_backward<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                    (apr_it, tree_it, level, z, x, patch_buffer, grad_patch, weights[ch_in][stencil_num],
                                     grad_output[batch], temp_dw, temp_vec, current_max_level, w_offset);

                            if (x >= stencilHalfX) {
                                /// grad_patch -> grad_input, grad_tree
                                for (int iz = z_start; iz < z_end; ++iz) {
                                    helpers::update_dense_array_backward<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                            (apr_it, tree_it, level, iz, x - stencilHalfX, temp_grad_input,
                                             grad_tree_local, grad_patch, current_max_level);
                                }
                            } else {
                                helpers::zero_boundary_x<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>(
                                        x, grad_patch);
                            }
                        } // x

                        /// grad_patch -> grad_input, grad_tree for boundary rows
                        for (int x = x_num - stencilHalfX; x < x_num; ++x) {
                            for (int iz = z_start; iz < z_end; ++iz) {
                                helpers::update_dense_array_backward<scalar_t, stencilSizeZ, stencilSizeX, stencilSizeY>
                                        (apr_it, tree_it, level, iz, x, temp_grad_input,
                                         grad_tree_local, grad_patch, current_max_level);
                            }
                        }
                    } // z
                } // level
#ifdef PYAPR_HAVE_OPENMP
#pragma omp critical(add_grad_tree)
#endif
                {
                    std::transform(grad_tree_local.begin(),
                                   grad_tree_local.end(),
                                   grad_tree.begin(),
                                   grad_tree.begin(),
                                   std::plus<scalar_t>{});
                }

#ifdef PYAPR_HAVE_OPENMP
#pragma omp critical(add_grad_input)
#endif
                {
                    std::transform(temp_grad_input.begin(),
                                   temp_grad_input.end(),
                                   grad_input[batch][ch_in].data(),
                                   grad_input[batch][ch_in].data(),
                                   std::plus<scalar_t>{});
                }
#ifdef PYAPR_HAVE_OPENMP
#pragma omp critical(add_grad_weights)
#endif
                {
                    std::transform(temp_dw.begin(),
                                   temp_dw.end(),
                                   grad_weights[ch_in].data(),
                                   grad_weights[ch_in].data(),
                                   std::plus<scalar_t>{});
                }
            } // parallel
            helpers::fill_tree_mean_backward<true>(apr, grad_input[batch][ch_in], grad_tree, current_max_level);
        } // ch_in
    } // batch
} // internal::backward

/// specialization for stencil size 1x1x1
template<>
template<typename scalar_t>
void Convolution<1, 1, 1>
                ::backward_impl(std::vector<APR*>& aprs,
                                torch::TensorAccessor<scalar_t, 3> input,
                                torch::TensorAccessor<scalar_t, 6> weights,
                                torch::TensorAccessor<scalar_t, 3> grad_input,
                                torch::TensorAccessor<scalar_t, 6> grad_weights,
                                torch::TensorAccessor<scalar_t, 3> grad_output,
                                torch::TensorAccessor<int, 1> level_deltas) {

    const int batch_size = input.size(0);
    const int number_in_channels = input.size(1);
    const int num_stencils = weights.size(1);
    const int number_out_channels = weights.size(2);

    for(int batch = 0; batch < batch_size; ++batch) {
        auto& apr = *aprs[batch];
        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();

        const int current_max_level = apr_it.level_max() - level_deltas[batch];

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for default(shared) firstprivate(apr_it, tree_it) if(number_in_channels > 15)
#endif
        for (int ch_in = 0; ch_in < number_in_channels; ++ch_in) {
            for (int ch_out = 0; ch_out < number_out_channels; ++ch_out) {
                for (int level = current_max_level; level >= apr_it.level_min(); --level) {
                    const int stencil_num = std::min(num_stencils - 1, current_max_level - level);
                    const scalar_t w = weights[ch_in][stencil_num][ch_out][0][0][0];
                    scalar_t d_weight = 0;

                    /// APR particles
                    const size_t l_begin = apr_it.particles_level_begin(level);
                    const size_t l_end = apr_it.particles_level_end(level);
                    d_weight += std::inner_product(input[batch][ch_in].data()+l_begin,
                                                   input[batch][ch_in].data()+l_end,
                                                   grad_output[batch][ch_out].data()+l_begin,
                                                   scalar_t(0));

                    std::transform(grad_output[batch][ch_out].data()+l_begin,
                                   grad_output[batch][ch_out].data()+l_end,
                                   grad_input[batch][ch_in].data()+l_begin,
                                   grad_input[batch][ch_in].data()+l_begin,
                                   [&w](scalar_t &d_o, scalar_t &d_i){ return d_i + w * d_o; });

                    /// downsampled particles
                    if (level == current_max_level && current_max_level < apr_it.level_max()) {
                        const auto tree_offset = helpers::compute_tree_offset(apr_it, tree_it, current_max_level);
                        const size_t l_begin_t = tree_it.particles_level_begin(current_max_level) + tree_offset;
                        const size_t l_end_t = tree_it.particles_level_end(current_max_level) + tree_offset;

                        d_weight += std::inner_product(input[batch][ch_in].data()+l_begin_t,
                                                       input[batch][ch_in].data()+l_end_t,
                                                       grad_output[batch][ch_out].data()+l_begin_t,
                                                       scalar_t(0));

                        std::transform(grad_output[batch][ch_out].data()+l_begin_t,
                                       grad_output[batch][ch_out].data()+l_end_t,
                                       grad_input[batch][ch_in].data()+l_begin_t,
                                       grad_input[batch][ch_in].data()+l_begin_t,
                                       [&w](scalar_t &d_o, scalar_t &d_i){ return d_i + w * d_o; });
                    }

                    grad_weights[ch_in][stencil_num][ch_out][0][0][0] += d_weight;
                }
            }
        }
    }
}

#endif //APRNET_CONV_HPP
