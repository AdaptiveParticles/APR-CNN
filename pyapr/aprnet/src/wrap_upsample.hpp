//
// Created by joel on 03.01.22.
//

#ifndef APRNET_WRAP_UPSAMPLE_HPP
#define APRNET_WRAP_UPSAMPLE_HPP

#include "upsample.hpp"
#include "wrapper.hpp"

void init_upsample(py::module &m) {
    using namespace pybind11::literals;

    m.def("upsample_const_forward", &UpSampleConst::forward, "aprs"_a, "input"_a, "level_delta"_a);
    m.def("upsample_const_backward", &UpSampleConst::backward, "aprs"_a, "grad_output"_a, "level_delta"_a);
}

#endif //APRNET_WRAP_UPSAMPLE_HPP
