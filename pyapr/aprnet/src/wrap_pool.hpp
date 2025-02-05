//
// Created by joel on 03.01.22.
//

#ifndef APRNET_WRAP_POOL_HPP
#define APRNET_WRAP_POOL_HPP

#include "maxpool.hpp"
#include "wrapper.hpp"

void init_pool(py::module &m) {
    using namespace pybind11::literals;

    m.def("maxpool_forward", MaxPool::forward, "aprs"_a, "input"_a, "level_delta"_a);
    m.def("maxpool_backward", &MaxPool::backward, "aprs"_a, "grad_output"_a, "max_indices"_a, "level_delta"_a);
}

#endif //APRNET_WRAP_POOL_HPP
