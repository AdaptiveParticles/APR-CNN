//
// Created by joel on 15.06.21.
//

#ifndef APRNET_UTILS_HPP
#define APRNET_UTILS_HPP

#include "wrapper.hpp"
#include "helpers.hpp"
#include <data_structures/APR/APR.hpp>
#include <data_structures/APR/particles/ParticleData.hpp>
#include <data_structures/Mesh/PixelData.hpp>
#include <cassert>

namespace utils {
    py::tuple find_nearest_particle(APR &apr, int z, int x, int y) {
        auto iterator = apr.iterator();

        if (z >= (int)apr.org_dims(2) || x >= (int)apr.org_dims(1) || y >= (int)apr.org_dims(0)) {
            throw std::invalid_argument("APRNet::utils::find_nearest_particle - queried coordinates out of bounds");
        }

        for (int level = iterator.level_min(); level <= iterator.level_max(); level++) {

            int z_lower = z / iterator.level_size(level);
            int x_lower = x / iterator.level_size(level);
            int y_lower = y / iterator.level_size(level);

            iterator.begin(level, z_lower, x_lower);
            while (iterator < iterator.end() && iterator.y() < y_lower) {
                iterator++;
            }

            if (iterator.y() == y_lower) {
                size_t idx = iterator;
                return py::make_tuple(idx, iterator.level_size(level));
            }
        }
        return py::make_tuple(-1, -1);
    }


    py::tuple get_coords_from_index(APR &apr, const size_t idx) {

        if (idx >= apr.total_number_particles()) {
            throw std::invalid_argument("APRNet::utils::get_coords_from_index - queried index is out of bounds");
        }

        auto iterator = apr.iterator();

        /// find level
        int level = iterator.level_min();
        while (iterator.particles_level_end(level) < idx) {
            level++;
        }

        /// find z
        int z = -1;
        do {
            z++;
            iterator.begin(level, z, iterator.x_num(level) - 1);
        } while (iterator.end() < idx);

        /// find x
        int x = -1;
        do {
            x++;
            iterator.begin(level, z, x);
        } while (iterator.end() < idx);

        /// find y
        iterator.begin(level, z, x);
        if (iterator > idx || iterator.end() < idx) {
            std::cerr << "something went wrong in APRNet::utils::get_coords_from_index. Searching for index " << idx
                      << " but have row begin " << iterator << " and end " << iterator.end() << std::endl;
            return py::make_tuple(level, z, x, -1);
        }
        while (iterator < idx) {
            iterator++;
        }
        int y = iterator.y();
        return py::make_tuple(level, z, x, y);
    }


    template<typename T>
    void labels_to_dist_cpp(APR &apr, py::array_t<T> &img, py::array_t<float> &dzs, py::array_t<float> &dxs,
                            py::array_t<float> &dys, py::array_t<float> &dist, const int level_delta) {

        auto imgc = img.template unchecked<3>();
        auto dzc = dzs.unchecked<1>();
        auto dxc = dxs.unchecked<1>();
        auto dyc = dys.unchecked<1>();
        auto distc = dist.mutable_unchecked<2>();

        auto it = apr.iterator();
        const int max_level = apr.level_max();
        const int current_max_level = max_level - level_delta;
        
        for (int level = it.level_min(); level <= current_max_level; level++) {
//#ifdef PYAPR_HAVE_OPENMP
//#pragma omp parallel for default(shared) schedule(dynamic) firstprivate(it)
//#endif
            for (int z = 0; z < it.z_num(level); z++) {
                for (int x = 0; x < it.x_num(level); x++) {
                    for (it.begin(level, z, x); it < it.end(); it++) {

                        const int y = it.y();

                        const int i = floor((z + 0.5) * (pow(2, (max_level - level))));
                        const int j = floor((x + 0.5) * (pow(2, (max_level - level))));
                        const int k = floor((y + 0.5) * (pow(2, (max_level - level))));

                        if (i >= imgc.shape(0) || j >= imgc.shape(1) || k >= imgc.shape(2)) {
                            continue;
                        }

                        const auto value = imgc(i, j, k);

                        if(value != 0){
                            float dz, dy, dx;
                            for (py::ssize_t n = 0; n < dzc.shape(0); n++) {
                                float xx = 0.0f;
                                float yy = 0.0f;
                                float zz = 0.0f;
                                dz = dzc(n);
                                dy = dyc(n);
                                dx = dxc(n);

                                while (true) {

                                    xx += dx;
                                    yy += dy;
                                    zz += dz;
                                    const int ii = lrint(i + zz);
                                    const int jj = lrint(j + xx);
                                    const int kk = lrint(k + yy);

                                    if((ii < 0) || (ii >= imgc.shape(0)) || 
                                       (jj < 0) || (jj >= imgc.shape(1)) ||
                                       (kk < 0) || (kk >= imgc.shape(2)) || 
                                       (value != imgc(ii, jj, kk))) {
                                        
                                        zz = lrint(zz);
                                        xx = lrint(xx);
                                        yy = lrint(yy);

                                        float dist = sqrt(xx*xx + yy*yy + zz*zz);
                                        distc(it, n) = dist;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if(level_delta > 0) {
            auto tree_it = apr.tree_iterator();
            const auto tree_offset = helpers::compute_tree_offset(it, tree_it, current_max_level);
            
            for(int z = 0; z < tree_it.z_num(current_max_level); ++z) {
                for(int x = 0; x < tree_it.x_num(current_max_level); ++x) {
                    for(tree_it.begin(current_max_level, z, x); tree_it < tree_it.end(); ++tree_it) {
                        
                        const int y = tree_it.y();

                        const int i = floor((z + 0.5) * (pow(2, (max_level - current_max_level))));
                        const int j = floor((x + 0.5) * (pow(2, (max_level - current_max_level))));
                        const int k = floor((y + 0.5) * (pow(2, (max_level - current_max_level))));

                        if (i >= imgc.shape(0) || j >= imgc.shape(1) || k >= imgc.shape(2)) {
                            continue;
                        }

                        const auto value = imgc(i, j, k);

                        if(value != 0){
                            float dz, dy, dx;
                            for (py::ssize_t n = 0; n < dzc.shape(0); n++) {
                                float xx = 0.0f;
                                float yy = 0.0f;
                                float zz = 0.0f;
                                dz = dzc(n);
                                dy = dyc(n);
                                dx = dxc(n);

                                while (true) {

                                    xx += dx;
                                    yy += dy;
                                    zz += dz;
                                    const int ii = lrint(i + zz);
                                    const int jj = lrint(j + xx);
                                    const int kk = lrint(k + yy);

                                    if((ii < 0) || (ii >= imgc.shape(0)) || 
                                       (jj < 0) || (jj >= imgc.shape(1)) ||
                                       (kk < 0) || (kk >= imgc.shape(2)) || 
                                       (value != imgc(ii, jj, kk))) {
                                        
                                        zz = lrint(zz);
                                        xx = lrint(xx);
                                        yy = lrint(yy);

                                        float dist = sqrt(xx*xx + yy*yy + zz*zz);
                                        distc(tree_it + tree_offset, n) = dist;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    void get_particle_coords(APR& apr, py::array_t<float>& coords, const int level_delta) {

        auto res = coords.mutable_unchecked<2>();
        assert(res.shape(0) == (int64_t) helpers::number_parts(apr, level_delta));
        assert(res.shape(1) == 3);

        auto it = apr.iterator();
        const int max_level = it.level_max();
        const int current_max_level = max_level - level_delta;

        for(int level = current_max_level; level >= it.level_min(); --level) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared) firstprivate(it)
#endif
            for(int z = 0; z < it.z_num(level); ++z) {
                for(int x = 0; x < it.x_num(level); ++x) {
                    for(it.begin(level, z, x); it < it.end(); ++it) {

                        const int y = it.y();

                        res(it, 0) = floor((z + 0.5) * pow(2, max_level - level));
                        res(it, 1) = floor((x + 0.5) * pow(2, max_level - level));
                        res(it, 2) = floor((y + 0.5) * pow(2, max_level - level));
                    }
                }
            }
        }

        if(level_delta > 0) {
            auto tree_it = apr.tree_iterator();
            const auto tree_offset = helpers::compute_tree_offset(it, tree_it, current_max_level);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared) firstprivate(tree_it)
#endif
            for(int z = 0; z < tree_it.z_num(current_max_level); ++z) {
                for(int x = 0; x < tree_it.x_num(current_max_level); ++x) {
                    for(tree_it.begin(current_max_level, z, x); tree_it < tree_it.end(); ++tree_it) {
                        const int y = tree_it.y();
                        res(tree_it + tree_offset, 0) = floor((z + 0.5) * pow(2, max_level - current_max_level));
                        res(tree_it + tree_offset, 1) = floor((x + 0.5) * pow(2, max_level - current_max_level));
                        res(tree_it + tree_offset, 2) = floor((y + 0.5) * pow(2, max_level - current_max_level));
                    }
                }
            }
        }
    }


    void sample_image(APR& apr, ParticleData<float>& parts, py::array_t<float, py::array::c_style>& img, const int level_delta){
        auto buf = img.request(false);
        auto shape = buf.shape;
        int z_num, x_num, y_num;
              
        if(buf.ndim == 3) {
            y_num = shape[2];
            x_num = shape[1];
            z_num = shape[0];
        } else if (buf.ndim == 2) {
            y_num = shape[1];
            x_num = shape[0];
            z_num = 1;
        } else if (buf.ndim == 1) {
            y_num = shape[0];
            x_num = 1;
            z_num = 1;
        } else {
            throw std::invalid_argument("sample_image: input array must be of dimension 1-3");
        }

        auto ptr = static_cast<float*>(buf.ptr);

        PixelData<float> pd;
        pd.init_from_mesh(y_num, x_num, z_num, ptr);
        
        //auto sum = [](const float x, const float y) -> float { return x + y; };
        //auto divide_by_8 = [](const float x) -> float { return x/8.0f; };
        
        auto sum = [](const float x, const float y) -> float { return std::max(x, y); };
        auto divide_by_8 = [](const float x) -> float { return x; };

        std::vector<PixelData<float>> image_pyramid;
        downsamplePyramid(pd, image_pyramid, apr.level_max(), apr.level_min(), sum, divide_by_8);
        
        auto apr_it = apr.iterator();
        const int current_max_level = apr_it.level_max() - level_delta;
        parts.init(helpers::number_parts(apr, level_delta));
        
        for(int level = apr_it.level_min(); level <= current_max_level; ++level) {
            for(int z = 0; z < apr_it.z_num(level); ++z) {
                for(int x = 0; x < apr_it.x_num(level); ++x) {
                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
                        parts[apr_it] = image_pyramid[level].at(apr_it.y(), x, z);
                    }
                }
            }
        }
        
        if(level_delta > 0) {
            auto tree_it = apr.tree_iterator();
            auto tree_offset = helpers::compute_tree_offset(apr_it, tree_it, current_max_level);
            
            for(int z = 0; z < tree_it.z_num(current_max_level); ++z) {
                for(int x = 0; x < tree_it.x_num(current_max_level); ++x) {
                    for(tree_it.begin(current_max_level, z, x); tree_it < tree_it.end(); ++tree_it) {
                        parts[tree_it + tree_offset] = image_pyramid[current_max_level].at(tree_it.y(), x, z);
                    }
                }
            }
        }
        
        std::swap(image_pyramid.back(), pd);
    }
}

void init_utils(py::module &m) {

    using namespace pybind11::literals;

    m.def("number_particles", &helpers::number_parts, "apr"_a, "level_delta"_a, R"pbdoc(
        Compute the number of particles at a certain level delta
    )pbdoc");

    m.def("number_particles_after_pooling", &helpers::number_parts_after_pooling, "apr"_a, "level_delta"_a, R"pbdoc(
        Compute the number of particles after downsampling the (current) highest resolution level
    )pbdoc");

    m.def("number_particles_after_upsampling", &helpers::number_parts_after_upsampling, "apr"_a, "level_delta"_a, R"pbdoc(
        Compute the number of particles after upsampling the (current) highest resolution level
    )pbdoc");

    m.def("labels_to_dist_cpp", &utils::labels_to_dist_cpp<uint8_t>, "apr"_a, "img"_a.noconvert(), "dz"_a.noconvert(), "dx"_a.noconvert(), 
        "dy"_a.noconvert(), "dist"_a.noconvert(), "level_delta"_a, R"pbdoc(
            Compute stardist radial distances for particles using a label image (pixels)
    )pbdoc");

    m.def("labels_to_dist_cpp", &utils::labels_to_dist_cpp<uint16_t>, "apr"_a, "img"_a.noconvert(), "dz"_a.noconvert(), "dx"_a.noconvert(), 
        "dy"_a.noconvert(), "dist"_a.noconvert(), "level_delta"_a, R"pbdoc(
            Compute stardist radial distances for particles using a label image (pixels)
    )pbdoc");

    m.def("labels_to_dist_cpp", &utils::labels_to_dist_cpp<uint32_t>, "apr"_a, "img"_a.noconvert(), "dz"_a.noconvert(), "dx"_a.noconvert(), 
        "dy"_a.noconvert(), "dist"_a.noconvert(), "level_delta"_a, R"pbdoc(
            Compute stardist radial distances for particles using a label image (pixels)
    )pbdoc");
    
    m.def("labels_to_dist_cpp", &utils::labels_to_dist_cpp<uint64_t>, "apr"_a, "img"_a.noconvert(), "dz"_a.noconvert(), "dx"_a.noconvert(), 
        "dy"_a.noconvert(), "dist"_a.noconvert(), "level_delta"_a, R"pbdoc(
            Compute stardist radial distances for particles using a label image (pixels)
    )pbdoc");
    
    m.def("get_particle_coords", &utils::get_particle_coords, "apr"_a, "coords"_a.noconvert(), "level_delta"_a, R"pbdoc(
        Find the (left/lower) center location (in pixel coordinates) of each particle
    )pbdoc");

    m.def("sample_image", &utils::sample_image, "apr"_a, "parts"_a, "img"_a.noconvert(), "level_delta"_a, R"pbdoc(
        Sample image with a level delta
    )pbdoc");

    m.def("number_parts", &helpers::number_parts, "apr"_a, "level_delta"_a, R"pbdoc(
        Number of particles at a given level delta
    )pbdoc");

}

#endif //APRNET_UTILS_HPP
