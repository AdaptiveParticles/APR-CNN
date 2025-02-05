//
// Created by joel on 04.01.22.
//

#ifndef APRNET_KERNEL_HPP
#define APRNET_KERNEL_HPP

#include <torch/all.h>

template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
class RestrictKernel {
public:
    static torch::Tensor forward(torch::Tensor weights,
                                 int num_levels);


    static torch::Tensor backward(torch::Tensor grad_weights,
                                  int num_stencils);

private:
    template<typename scalar_t>
    static void forward_impl(torch::TensorAccessor<scalar_t, 6> input,
                             torch::TensorAccessor<scalar_t, 6> output);

    template<typename scalar_t>
    static void backward_impl(torch::TensorAccessor<scalar_t, 6> grad_output,
                              torch::TensorAccessor<scalar_t, 6> grad_input);
};


template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
torch::Tensor RestrictKernel<stencilSizeZ, stencilSizeX, stencilSizeY>
        ::forward(torch::Tensor weights,
                  const int num_levels) {

    auto output = torch::zeros({weights.size(0),
                                num_levels,
                                weights.size(2),
                                weights.size(3),
                                weights.size(4),
                                weights.size(5)}, weights.dtype());

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "RestrictKernel::forward dispatch", ([&] {
        RestrictKernel<stencilSizeZ, stencilSizeX, stencilSizeY>::forward_impl<scalar_t>(
                weights.accessor<scalar_t, 6>(),
                output.accessor<scalar_t, 6>());
    }));

    return output;
}


template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
template<typename scalar_t>
void RestrictKernel<stencilSizeZ, stencilSizeX, stencilSizeY>::forward_impl(
        torch::TensorAccessor<scalar_t, 6> input,
        torch::TensorAccessor<scalar_t, 6> output) {

    const int ndim = (stencilSizeZ > 1) + (stencilSizeX > 1) + (stencilSizeY > 1);

    const int in_channels = input.size(0);
    const int num_stencils_in = input.size(1);
    const int num_stencils_out = output.size(1);
    const int out_channels = input.size(2);

    const size_t filter_bank_size = out_channels * stencilSizeZ * stencilSizeX * stencilSizeY;
    const int source_level = num_stencils_in - 1;

    for(int ch_in = 0; ch_in < in_channels; ++ch_in) {

        for(int level = 0; level < num_stencils_in; ++level) {
            std::copy(input[ch_in][level].data(),
                      input[ch_in][level].data() + filter_bank_size,
                      output[ch_in][level].data());
        }

        for (int level = num_stencils_in; level < num_stencils_out; ++level) {
            const int level_delta = level - source_level;
            const int step_size = (int) std::pow(2.0f, (float) level_delta);
            const float factor = std::pow((float) step_size, (float) ndim);

            const int z_offset = ((stencilSizeZ / 2) * step_size - stencilSizeZ / 2);
            const int x_offset = ((stencilSizeX / 2) * step_size - stencilSizeX / 2);
            const int y_offset = ((stencilSizeY / 2) * step_size - stencilSizeY / 2);

            for (int z_in = 0; z_in < stencilSizeZ; ++z_in) {
                for (int x_in = 0; x_in < stencilSizeX; ++x_in) {
                    for (int y_in = 0; y_in < stencilSizeY; ++y_in) {

                        for (int z_out = 0; z_out < stencilSizeZ; ++z_out) {

                            const int z_start = std::max(z_out * step_size, z_offset + z_in);
                            const int z_end = std::min((z_out + 1) * step_size, z_offset + z_in + step_size);
                            const float overlap_z = z_end - z_start;

                            if (overlap_z <= 0) { continue; }

                            for (int x_out = 0; x_out < stencilSizeX; ++x_out) {

                                const int x_start = std::max(x_out * step_size, x_offset + x_in);
                                const int x_end = std::min((x_out + 1) * step_size, x_offset + x_in + step_size);
                                const float overlap_x = std::max(x_end - x_start, 0);

                                if (overlap_x <= 0) { continue; }

                                for (int y_out = 0; y_out < stencilSizeY; ++y_out) {

                                    const int y_start = std::max(y_out * step_size, y_offset + y_in);
                                    const int y_end = std::min((y_out + 1) * step_size, y_offset + y_in + step_size);
                                    const float overlap_y = std::max(y_end - y_start, 0);

                                    if (overlap_y <= 0) { continue; }

                                    const float overlap = overlap_x * overlap_y * overlap_z / factor;
                                    for(int ch_out = 0; ch_out < out_channels; ++ch_out) {
                                        output[ch_in][level][ch_out][z_out][x_out][y_out] += overlap * input[ch_in][source_level][ch_out][z_in][x_in][y_in];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
torch::Tensor RestrictKernel<stencilSizeZ, stencilSizeX, stencilSizeY>
        ::backward(torch::Tensor grad_weights,
                   const int num_stencils) {

    auto output = torch::zeros({grad_weights.size(0),
                                num_stencils,
                                grad_weights.size(2),
                                grad_weights.size(3),
                                grad_weights.size(4),
                                grad_weights.size(5)}, grad_weights.dtype());

    AT_DISPATCH_FLOATING_TYPES(grad_weights.scalar_type(), "RestrictKernel::backward dispatch", ([&] {
        RestrictKernel<stencilSizeZ, stencilSizeX, stencilSizeY>::backward_impl<scalar_t>(
                grad_weights.accessor<scalar_t, 6>(),
                output.accessor<scalar_t, 6>());
    }));

    return output;
}


template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
template<typename scalar_t>
void RestrictKernel<stencilSizeZ, stencilSizeX, stencilSizeY>::backward_impl(
        torch::TensorAccessor<scalar_t, 6> grad_output,
        torch::TensorAccessor<scalar_t, 6> grad_input) {

    const int ndim = (stencilSizeZ > 1) + (stencilSizeX > 1) + (stencilSizeY > 1);

    const int in_channels = grad_input.size(0);
    const int num_stencils_in = grad_input.size(1);
    const int num_stencils_out = grad_output.size(1);
    const int out_channels = grad_input.size(2);

    const size_t filter_bank_size = out_channels * stencilSizeZ * stencilSizeX * stencilSizeY;
    const int source_level = num_stencils_in - 1;

    for(int ch_in = 0; ch_in < in_channels; ++ch_in) {

        for(int level = 0; level < num_stencils_in; ++level) {
            std::copy(grad_output[ch_in][level].data(),
                      grad_output[ch_in][level].data() + filter_bank_size,
                      grad_input[ch_in][level].data());
        }

        for (int level = num_stencils_in; level < num_stencils_out; ++level) {
            const int level_delta = level - source_level;
            const int step_size = (int) std::pow(2.0f, (float) level_delta);
            const float factor = std::pow((float) step_size, (float) ndim);

            const int z_offset = ((stencilSizeZ / 2) * step_size - stencilSizeZ / 2);
            const int x_offset = ((stencilSizeX / 2) * step_size - stencilSizeX / 2);
            const int y_offset = ((stencilSizeY / 2) * step_size - stencilSizeY / 2);

            for (int z_in = 0; z_in < stencilSizeZ; ++z_in) {
                for (int x_in = 0; x_in < stencilSizeX; ++x_in) {
                    for (int y_in = 0; y_in < stencilSizeY; ++y_in) {

                        for (int z_out = 0; z_out < stencilSizeZ; ++z_out) {

                            const int z_start = std::max(z_out * step_size, z_offset + z_in);
                            const int z_end = std::min((z_out + 1) * step_size, z_offset + z_in + step_size);
                            const float overlap_z = z_end - z_start;

                            if (overlap_z <= 0) { continue; }

                            for (int x_out = 0; x_out < stencilSizeX; ++x_out) {

                                const int x_start = std::max(x_out * step_size, x_offset + x_in);
                                const int x_end = std::min((x_out + 1) * step_size, x_offset + x_in + step_size);
                                const float overlap_x = std::max(x_end - x_start, 0);

                                if (overlap_x <= 0) { continue; }

                                for (int y_out = 0; y_out < stencilSizeY; ++y_out) {

                                    const int y_start = std::max(y_out * step_size, y_offset + y_in);
                                    const int y_end = std::min((y_out + 1) * step_size, y_offset + y_in + step_size);
                                    const float overlap_y = std::max(y_end - y_start, 0);

                                    if (overlap_y <= 0) { continue; }

                                    const float overlap = overlap_x * overlap_y * overlap_z / factor;
                                    for(int ch_out = 0; ch_out < out_channels; ++ch_out) {
                                        grad_input[ch_in][source_level][ch_out][z_in][x_in][y_in] += overlap * grad_output[ch_in][level][ch_out][z_out][x_out][y_out];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif //APRNET_KERNEL_HPP
