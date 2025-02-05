//
// Created by Joel Jonsson on 29.06.18.
//

#ifndef PYLIBAPR_PYAPR_HPP
#define PYLIBAPR_PYAPR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include <data_structures/APR/APR.hpp>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<APR*>)


template<typename T>
py::array_t<T> vd_to_array(VectorData<T>& vec) {
    py::array_t<T> arr(vec.size());
    auto r = arr.template mutable_unchecked<1>();
    for(size_t i = 0; i < vec.size(); ++i) {
        r(i) = vec[i];
    }
    return arr;
}

template<typename T>
void array_to_vd(py::array_t<T> arr, VectorData<T>& vec) {
    auto r = arr.template unchecked<1>();
    vec.resize(r.shape(0));
    for(py::ssize_t i = 0; i < r.shape(0); ++i) {
        vec[i] = r(i);
    }
}


void AddAPR(pybind11::module &m, const std::string &modulename) {

    using namespace py::literals;

    py::class_<APR>(m, modulename.c_str())
            .def(py::init())
            .def("__repr__", [](APR& a) {
                return "APR(shape [" + std::to_string(a.org_dims(2)) + ", " + std::to_string(a.org_dims(1)) +
                        ", " + std::to_string(a.org_dims(0)) + "], " + std::to_string(a.total_number_particles()) + " particles)";})
            .def_readwrite("name", &APR::name)
            .def("total_number_particles", &APR::total_number_particles, "return number of particles")
            .def("total_number_tree_particles", &APR::total_number_tree_particles, "return number of interior tree particles")
            .def("level_min", &APR::level_min, "return the minimum resolution level")
            .def("level_max", &APR::level_max, "return the maximum resolution level")
            .def("x_num", &APR::x_num,  "Gives the maximum bounds in the x direction for the given level", "level"_a)
            .def("y_num", &APR::y_num,  "Gives the maximum bounds in the y direction for the given level", "level"_a)
            .def("z_num", &APR::z_num,  "Gives the maximum bounds in the z direction for the given level", "level"_a)
            .def("iterator", &APR::iterator, "Return a linear iterator for APR particles")
            .def("tree_iterator", &APR::tree_iterator, "Return a linear iterator for interior APRTree particles")
            .def("org_dims", &APR::org_dims, "returns the original image size in a specified dimension (y, x, z)" , "dim"_a)
            .def("shape", [](APR& self){return py::make_tuple(self.org_dims(2), self.org_dims(1), self.org_dims(0));}, "returns the original pixel image dimensions as a tuple (z, x, y)")
            .def("get_parameters", &APR::get_apr_parameters, "return the parameters used to create the APR")
#ifdef PYAPR_USE_CUDA
            .def("init_cuda", &APR::init_cuda, "copy access structures to GPU device", "with_tree"_a=true)
#endif
            .def("computational_ratio", &APR::computational_ratio, "return the computational ratio (number of pixels in original image / number of particles in the APR)")
            .def(py::pickle(
                [](APR &apr) { // __getstate__
                    if(!apr.apr_initialized) {
                        throw std::runtime_error("Invalid state - APR not initialized");
                    }

                    if(apr.tree_initialized) {  // include tree access vectors
                        return py::make_tuple(
                                    apr.org_dims(0), 
                                    apr.org_dims(1), 
                                    apr.org_dims(2),
                                    apr.parameters,
                                    vd_to_array<uint64_t>(apr.linearAccess.xz_end_vec),
                                    vd_to_array<uint16_t>(apr.linearAccess.y_vec),
                                    vd_to_array<uint64_t>(apr.linearAccessTree.xz_end_vec),
                                    vd_to_array<uint16_t>(apr.linearAccessTree.y_vec));
                    } else {
                        return py::make_tuple(
                                    apr.org_dims(0), 
                                    apr.org_dims(1), 
                                    apr.org_dims(2),
                                    apr.parameters,
                                    vd_to_array<uint64_t>(apr.linearAccess.xz_end_vec),
                                    vd_to_array<uint16_t>(apr.linearAccess.y_vec));
                    }
                },
                [](py::tuple t) { // __setstate__
                    if (t.size() != 6 && t.size() != 8)
                        throw std::runtime_error("Error unpickling APR object - invalid state size");

                    APR apr;
                    
                    // initialize GenInfo
                    apr.aprInfo.init(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>());
                    apr.treeInfo.init_tree(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>());

                    apr.parameters = t[3].cast<APRParameters>();
                    
                    // initialize LinearAccess
                    apr.linearAccess.genInfo = &apr.aprInfo;
                    apr.linearAccessTree.genInfo = &apr.treeInfo;
                    apr.linearAccess.initialize_xz_linear();
                    apr.linearAccessTree.initialize_xz_linear();

#ifdef PYAPR_USE_CUDA
                    apr.gpuAccess.genInfo = &apr.aprInfo;
                    apr.gpuTreeAccess.genInfo = &apr.treeInfo;
#endif

                    // copy APR access data
                    array_to_vd(t[4].cast<py::array_t<uint64_t>>(), apr.linearAccess.xz_end_vec);
                    array_to_vd(t[5].cast<py::array_t<uint16_t>>(), apr.linearAccess.y_vec);
                    apr.aprInfo.total_number_particles = apr.linearAccess.y_vec.size();
                    apr.apr_initialized = true;

                    // copy tree access data if available
                    if(t.size() == 8) {
                        array_to_vd(t[6].cast<py::array_t<uint64_t>>(), apr.linearAccessTree.xz_end_vec);
                        array_to_vd(t[7].cast<py::array_t<uint16_t>>(), apr.linearAccessTree.y_vec);
                        apr.treeInfo.total_number_particles = apr.linearAccessTree.y_vec.size();
                        apr.tree_initialized = true;
                    }
                    
                    return apr;
                })
            );

    py::bind_vector<std::vector<APR*>>(m, "APRPtrVector", py::module_local(false));
}


#endif //PYLIBAPR_PYAPR_HPP
