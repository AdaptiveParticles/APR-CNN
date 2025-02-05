//
// Created by joel on 03.01.22.
//

#ifndef APRNET_WRAP_CONV_HPP
#define APRNET_WRAP_CONV_HPP

#include "conv.hpp"
#include "kernel.hpp"
#include "wrapper.hpp"

template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
void bind_convolution(py::module &m) {
    using namespace pybind11::literals;
    std::string name = "convolve" + std::to_string(stencilSizeZ) + std::to_string(stencilSizeX) + std::to_string(stencilSizeY);

    m.def((name + "_forward").c_str(), &Convolution<stencilSizeZ, stencilSizeX, stencilSizeY>::forward,
          "aprs"_a, "input"_a, "weights"_a, "level_deltas"_a);

    m.def((name + "_backward").c_str(), &Convolution<stencilSizeZ, stencilSizeX, stencilSizeY>::backward,
          "aprs"_a, "input"_a, "weights"_a, "grad_output"_a, "level_deltas"_a);
}

template<int stencilSizeZ, int stencilSizeX, int stencilSizeY>
void bind_restrict(py::module &m) {
    using namespace pybind11::literals;
    std::string name = "restrict_kernel" + std::to_string(stencilSizeZ) + std::to_string(stencilSizeX) + std::to_string(stencilSizeY);

    m.def((name + "_forward").c_str(), &RestrictKernel<stencilSizeZ, stencilSizeX, stencilSizeY>::forward,
          "weights"_a, "num_levels"_a);

    m.def((name + "_backward").c_str(), &RestrictKernel<stencilSizeZ, stencilSizeX, stencilSizeY>::backward,
          "grad_weights"_a, "num_stencils"_a);
}


void init_conv(py::module &m) {

    bind_convolution<1, 1, 1>(m);
    bind_convolution<3, 3, 3>(m);
    bind_convolution<1, 3, 3>(m);
    bind_convolution<5, 5, 5>(m);
    bind_convolution<1, 5, 5>(m);

    bind_restrict<3, 3, 3>(m);
    bind_restrict<1, 3, 3>(m);
    bind_restrict<5, 5, 5>(m);
    bind_restrict<1, 5, 5>(m);
}

#endif //APRNET_WRAP_CONV_HPP
