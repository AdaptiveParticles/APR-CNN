#include <ConfigAPR.h>
#include <pybind11/pybind11.h>

#include "converter/src/BindConverter.hpp"
#include "converter/src/BindConverterBatch.hpp"

#include "data_containers/src/BindPixelData.hpp"
#include "data_containers/src/BindAPR.hpp"
#include "data_containers/src/BindParameters.hpp"
#include "data_containers/src/BindParticleData.hpp"
#include "data_containers/src/BindReconPatch.hpp"
#include "data_containers/src/BindLinearIterator.hpp"
#include "data_containers/src/BindLazyAccess.hpp"
#include "data_containers/src/BindLazyData.hpp"
#include "data_containers/src/BindLazyIterator.hpp"

#include "filter/src/BindFilter.hpp"
#include "io/src/BindAPRFile.hpp"
#include "measure/src/BindMeasure.hpp"
#include "morphology/src/BindMorphology.hpp"
#include "reconstruction/src/BindReconstruction.hpp"
#include "restoration/src/BindRichardsonLucy.hpp"
#include "segmentation/src/BindGraphCut.hpp"
#include "transform/src/BindProjection.hpp"
#include "tree/src/BindFillTree.hpp"

#include "viewer/src/BindRaycaster.hpp"
#include "viewer/src/BindViewerHelpers.hpp"

#ifdef BUILD_APRNET
#include "aprnet/src/utils.hpp"
#include "aprnet/src/wrap_conv.hpp"
#include "aprnet/src/wrap_pool.hpp"
#include "aprnet/src/wrap_upsample.hpp"

#ifdef PYAPR_USE_CUDA
#include "aprnet/cuda/src/wrap_conv_gpu.hpp"
#include "aprnet/cuda/src/wrap_maxpool_gpu.hpp"
#include "aprnet/cuda/src/wrap_upsample_const_gpu.hpp"
#include "aprnet/cuda/src/wrap_restrict_gpu.hpp"
#include "aprnet/cuda/src/wrap_reconstruct_gpu.hpp"
#include "aprnet/cuda/src/wrap_sample_particles_gpu.hpp"
#endif

#endif


#ifdef PYAPR_USE_CUDA
#define BUILT_WITH_CUDA true
#else
#define BUILT_WITH_CUDA false
#endif

namespace py = pybind11;
using namespace pybind11::literals;

// -------- Check if properly configured in CMAKE -----------------------------
#ifndef APR_PYTHON_MODULE_NAME
#error "Name of APR module (python binding) is not defined!"
#endif

// -------- Definition of python module ---------------------------------------
PYBIND11_MODULE(APR_PYTHON_MODULE_NAME, m) {
    m.doc() = "python binding for LibAPR library";
    m.attr("__version__") = py::str(ConfigAPR::APR_VERSION);
    m.attr("__cuda_build__") = BUILT_WITH_CUDA;

    py::module data_containers = m.def_submodule("data_containers");

    AddAPR(data_containers, "APR");
    AddAPRParameters(data_containers);
    AddLinearIterator(data_containers);
    AddReconPatch(data_containers);

    AddPyPixelData<uint8_t>(data_containers, "Byte");
    AddPyPixelData<uint16_t>(data_containers, "Short");
    AddPyPixelData<float>(data_containers, "Float");
    AddPyPixelData<uint64_t>(data_containers, "Long");

    AddPyParticleData<uint8_t>(data_containers, "Byte");
    AddPyParticleData<float>(data_containers, "Float");
    AddPyParticleData<uint16_t>(data_containers, "Short");
    AddPyParticleData<uint64_t>(data_containers, "Long");

    AddLazyAccess(data_containers, "LazyAccess");
    AddLazyIterator(data_containers);
    AddLazyData<uint8_t>(data_containers, "Byte");
    AddLazyData<uint16_t>(data_containers, "Short");
    AddLazyData<uint64_t>(data_containers, "Long");
    AddLazyData<float>(data_containers, "Float");


    py::module converter = m.def_submodule("converter");

    AddPyAPRConverter<uint8_t>(converter, "Byte");
    AddPyAPRConverter<uint16_t>(converter, "Short");
    AddPyAPRConverter<float>(converter, "Float");

    AddPyAPRConverterBatch<uint8_t>(converter, "Byte");
    AddPyAPRConverterBatch<uint16_t>(converter, "Short");
    AddPyAPRConverterBatch<float>(converter, "Float");


    py::module filter = m.def_submodule("filter");
    AddFilter(filter);


    py::module io = m.def_submodule("io");
    AddAPRFile(io, "APRFile");


    py::module measure = m.def_submodule("measure");
    AddMeasure(measure);


    py::module morphology = m.def_submodule("morphology");
    AddMorphology(morphology);


    py::module reconstruction = m.def_submodule("reconstruction");
    AddReconstruction(reconstruction);


    py::module restoration = m.def_submodule("restoration");
    AddRichardsonLucy(restoration);


    py::module segmentation = m.def_submodule("segmentation");
    AddGraphcut(segmentation, "graphcut");


    py::module transform = m.def_submodule("transform");
    AddProjection(transform);


    py::module tree = m.def_submodule("tree");
    AddFillTree(tree);


    py::module viewer = m.def_submodule("viewer");
    AddViewerHelpers(viewer);
    AddRaycaster(viewer, "APRRaycaster");

#ifdef BUILD_APRNET
    py::module aprnet = m.def_submodule("aprnet");

    py::module utils_module = aprnet.def_submodule("utils");
    init_utils(utils_module);

    py::module conv_module = aprnet.def_submodule("conv");
    init_conv(conv_module);

    py::module pool_module = aprnet.def_submodule("pool");
    init_pool(pool_module);

    py::module upsample_module = aprnet.def_submodule("upsample");
    init_upsample(upsample_module);

#ifdef PYAPR_USE_CUDA
    py::module cuda_module = aprnet.def_submodule("cuda");
    
    cuda_module.def("maxpool_forward_cuda", &maxpool_forward_cuda, "aprs"_a, "input"_a, "level_delta"_a);
    cuda_module.def("maxpool_backward_cuda", &maxpool_backward_cuda, "aprs"_a, "grad_output"_a, "max_indices"_a, "level_delta"_a);

    cuda_module.def("conv111_forward", &conv111_forward, "aprs"_a, "input"_a, "weights"_a, "level_delta"_a);
    cuda_module.def("conv111_backward", &conv111_backward, "aprs"_a, "input"_a, "weights"_a, "grad_output"_a, "level_delta"_a);

    cuda_module.def("conv333_forward", &conv333_forward, "aprs"_a, "input"_a, "weights"_a, "level_delta"_a);
    cuda_module.def("conv333_backward", &conv333_backward, "aprs"_a, "input"_a, "weights"_a, "grad_output"_a, "level_delta"_a);

    cuda_module.def("upsample_const_forward", &upsample_const_forward, "aprs"_a, "input"_a, "level_delta"_a);
    cuda_module.def("upsample_const_backward", &upsample_const_backward, "aprs"_a, "grad_output"_a, "level_delta"_a);

    cuda_module.def("restrict_kernel333_forward", &restrict_kernel_333_forward, "weights"_a, "num_levels"_a);
    cuda_module.def("restrict_kernel333_backward", &restrict_kernel_333_backward, "grad_weights_expanded"_a, "num_levels"_a);

    cuda_module.def("reconstruct_forward", &reconstruct_forward, "aprs"_a, "input"_a, "level_delta"_a);
    cuda_module.def("reconstruct_backward", &reconstruct_backward, "aprs"_a, "grad_output"_a, "level_delta"_a);
	
    cuda_module.def("sample_particles_forward", &sample_particles_forward, "aprs"_a, "input"_a, "level_delta"_a);
    cuda_module.def("sample_particles_backward", &sample_particles_backward, "aprs"_a, "grad_output"_a, "level_delta"_a);
#endif
#endif
}
