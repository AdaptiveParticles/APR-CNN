#ifndef PYLIBAPR_PYAPRPARAMETERS_HPP
#define PYLIBAPR_PYAPRPARAMETERS_HPP

#include <algorithm/APRParameters.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;


void AddAPRParameters(pybind11::module &m) {

    using namespace py::literals;

    py::class_<APRParameters>(m, "APRParameters")
        .def(py::init())
        // Commonly used parameters
        .def_readwrite("dz", &APRParameters::dz, "Voxel size in dimension z (shape[0] of image array). Default 1.0")
        .def_readwrite("dx", &APRParameters::dx, "Voxel size in dimension x (shape[1] of image array). Default 1.0")
        .def_readwrite("dy", &APRParameters::dy, "Voxel size in dimension y (shape[2] of image array). Default 1.0")
        .def_readwrite("gradient_smoothing", &APRParameters::lambda,
                       "Degree of smoothing in the B-spline fit used in internal steps "
                       "(0 -> no smoothing, higher -> more smoothing). Default 3.0")
        .def_readwrite("rel_error", &APRParameters::rel_error, "Relative error threshold of the reconstruction condition")
        .def_readwrite("Ip_th", &APRParameters::Ip_th, "Regions below this intensity are ignored. Default 0")
        .def_readwrite("sigma_th", &APRParameters::sigma_th, "The local intensity scale is clamped from below to this value")
        .def_readwrite("grad_th", &APRParameters::grad_th, "Gradients below this value are set to 0")
        .def_readwrite("auto_parameters", &APRParameters::auto_parameters,
                       "If True, compute sigma_th and grad_th using minimum cross entropy thresholding (Li's algorithm). Default False")

        // Debugging and analysis
        .def_readwrite("output_steps", &APRParameters::output_steps,
                       "If True, intermediate steps are saved as tiff files in the directory specified by output_dir. Default False")
        .def_readwrite("output_dir", &APRParameters::output_dir,
                       "Output directory for intermediate steps (only used if output_steps is True)")

        // Additional parameters (can usually be kept at default values)
        .def_readwrite("psfx", &APRParameters::psfx, "Affects local intensity scale window size")
        .def_readwrite("psfy", &APRParameters::psfy, "Affects local intensity scale window size")
        .def_readwrite("psfz", &APRParameters::psfz, "Affects local intensity scale window size")
        .def_readwrite("neighborhood_optimization", &APRParameters::neighborhood_optimization)
        .def_readwrite("sigma_th_max", &APRParameters::sigma_th_max)
        .def_readwrite("noise_sd_estimate", &APRParameters::noise_sd_estimate)
        .def_readwrite("background_intensity_estimate", &APRParameters::background_intensity_estimate)
        .def_readwrite("name", &APRParameters::name, "Name of the APR")
        .def_readwrite("input_image_name", &APRParameters::input_image_name)
        .def_readwrite("input_dir", &APRParameters::input_dir)
        .def_readwrite("mask_file", &APRParameters::mask_file)
        .def(py::pickle(
                [](APRParameters &par) { // __getstate__
                    return py::make_tuple(
                        par.dz,
                        par.dx,
                        par.dy,
                        par.lambda,
                        par.rel_error,
                        par.Ip_th,
                        par.sigma_th,
                        par.grad_th,
                        par.auto_parameters,
                        par.psfx,
                        par.psfy,
                        par.psfz,
                        par.neighborhood_optimization,
                        par.sigma_th_max,
                        par.noise_sd_estimate,
                        par.background_intensity_estimate,
                        par.name,
                        par.input_image_name,
                        par.input_dir,
                        par.mask_file,
                        par.output_steps,
                        par.output_dir
                    );
                },
                [](py::tuple t) {   // __setstate__
                    if (t.size() != 22)
                        throw std::runtime_error("Error unpickling APRParameters object - invalid state size");
                    APRParameters par;
                    par.dz = t[0].cast<float>();
                    par.dx = t[1].cast<float>();
                    par.dy = t[2].cast<float>();
                    par.lambda = t[3].cast<float>();
                    par.rel_error = t[4].cast<float>();
                    par.Ip_th = t[5].cast<float>();
                    par.sigma_th = t[6].cast<float>();
                    par.grad_th = t[7].cast<float>();
                    par.auto_parameters = t[8].cast<bool>();
                    par.psfx = t[9].cast<float>();
                    par.psfy = t[10].cast<float>();
                    par.psfz = t[11].cast<float>();
                    par.neighborhood_optimization = t[12].cast<bool>();
                    par.sigma_th_max = t[13].cast<float>();
                    par.noise_sd_estimate = t[14].cast<float>();
                    par.background_intensity_estimate = t[15].cast<float>();
                    par.name = t[16].cast<std::string>();
                    par.input_image_name = t[17].cast<std::string>();
                    par.input_dir = t[18].cast<std::string>();
                    par.mask_file = t[19].cast<std::string>();
                    par.output_steps = t[20].cast<bool>();
                    par.output_dir = t[21].cast<std::string>();
                    return par;
                }
            )
        );
}

#endif //PYLIBAPR_PYAPRPARAMETERS_HPP
