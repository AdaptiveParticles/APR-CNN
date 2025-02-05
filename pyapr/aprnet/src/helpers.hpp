//
// Created by joel on 05.07.21.
//

#ifndef APRNET_HELPERS_HPP
#define APRNET_HELPERS_HPP

#include <torch/all.h>
#include "APRNetImageBuffer.hpp"
#include <data_structures/APR/APR.hpp>
#include <data_structures/APR/particles/ParticleData.hpp>

namespace helpers {

    template<bool parallel, typename scalar_t>
    void fill_tree_mean(APR &apr,
                        const torch::TensorAccessor<scalar_t, 1> input,
                        ParticleData<scalar_t> &tree_data,
                        const int current_max_level);

    template<bool parallel, typename scalar_t>
    void fill_tree_mean_backward(APR &apr,
                                 torch::TensorAccessor<scalar_t, 1> grad_input,
                                 ParticleData<scalar_t> &grad_tree,
                                 const int current_max_level);

    constexpr int compute_number_parent_levels(int sz, int sx, int sy);

    template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
    void update_dense_array(LinearIterator &apr_it,
                            LinearIterator &tree_it,
                            const int level,
                            const int z,
                            const int x,
                            const torch::TensorAccessor<scalar_t, 1> input,
                            const ParticleData<scalar_t> &tree_data,
                            APRNetImageBuffer<scalar_t> &patch_buffer,
                            const int current_max_level);

    template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
    void update_dense_array_backward(LinearIterator &apr_it,
                                     LinearIterator &tree_it,
                                     const int level,
                                     const int z,
                                     const int x,
                                     std::vector<scalar_t> &grad_input,
                                     ParticleData<scalar_t> &grad_tree,
                                     const APRNetImageBuffer<scalar_t> &grad_patch,
                                     const int current_max_level);

    template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
    inline void zero_boundary_z(const int z, APRNetImageBuffer<scalar_t> &patch_buffer);

    template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
    inline void zero_boundary_x(const int x, APRNetImageBuffer<scalar_t> &patch_buffer);


    template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
    void accumulate_convolution(LinearIterator& apr_it,
                                LinearIterator& tree_it,
                                const int level,
                                const int z,
                                const int x,
                                const APRNetImageBuffer<scalar_t>& patch,
                                const torch::TensorAccessor<scalar_t, 4> filter,
                                torch::TensorAccessor<scalar_t, 2> output,
                                std::vector<scalar_t>& temp_vec,
                                const int current_max_level);


    template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
    void accumulate_convolution_backward(LinearIterator &apr_it,
                                         LinearIterator &tree_it,
                                         const int level,
                                         const int z,
                                         const int x,
                                         const APRNetImageBuffer<scalar_t> &patch,
                                         APRNetImageBuffer<scalar_t> &grad_patch,
                                         const torch::TensorAccessor<scalar_t, 4> filter,
                                         const torch::TensorAccessor<scalar_t, 2> grad_output,
                                         std::vector<scalar_t> &temp_dw,
                                         std::vector<scalar_t> &temp_vec,
                                         const int current_max_level,
                                         const size_t w_offset);

    size_t number_parts(APR& apr, const int level_delta) {
        auto apr_it = apr.iterator();
        auto tree_it = apr.tree_iterator();

        const int current_max_level = apr.level_max() - level_delta;
        size_t number_parts = 0, tree_start = 0, tree_end = 0;

        number_parts = apr_it.particles_level_end(current_max_level);

        if(current_max_level <= tree_it.level_max()) {
            tree_start = tree_it.particles_level_begin(current_max_level);
            tree_end = tree_it.particles_level_end(current_max_level);
        }

        return (number_parts + tree_end) - tree_start;
    }

    inline size_t number_parts_after_pooling(APR &apr, const int level_delta) {
        return number_parts(apr, level_delta+1);
    }


    inline size_t number_parts_after_upsampling(APR &apr, int level_delta) {
        return number_parts(apr, level_delta-1);
    }


    inline int64_t compute_tree_offset(LinearIterator &apr_iterator, LinearIterator &tree_iterator, const int level){
        int64_t number_parts = apr_iterator.particles_level_end(level);
        int64_t tree_start = tree_iterator.particles_level_begin(level);
        return number_parts - tree_start;
    }

    int min_occupied_level(APR& apr) {
        auto it = apr.iterator();
        for(int level = 1; level <= it.level_max(); ++level) {
            it.begin(level, it.z_num(level)-1, it.x_num(level)-1);
            if(it.end() > 0) { 
                return level; 
            }
        }
        return 1;
    }

}


template<class Iterator1, class Iterator2>
inline float _compute_scale_factor_xz(Iterator1& child_iterator,
                                      Iterator2& parent_iterator,
                                      const int level,
                                      const int z,
                                      const int x) {
    float scale_factor_xz =
                (((2 * parent_iterator.x_num(level - 1)) != child_iterator.x_num(level)) &&
                ((x / 2) == (parent_iterator.x_num(level - 1) - 1))) +
                (((2 * parent_iterator.z_num(level - 1)) != child_iterator.z_num(level)) &&
                ((z / 2) == (parent_iterator.z_num(level - 1) - 1)));
    return std::max(scale_factor_xz * 2.0f, 1.0f) / 8.0f;
}

template<class Iterator1, class Iterator2>
inline float _compute_scale_factor_yxz(const float scale_factor_xz,
                                       Iterator1& child_iterator,
                                       Iterator2& parent_iterator,
                                       const int level) {
    return ((2 * parent_iterator.y_num(level - 1)) != child_iterator.y_num(level)) ? scale_factor_xz * 2.0f : scale_factor_xz;
}


template<bool parallel, typename scalar_t>
void helpers::fill_tree_mean(APR &apr,
                             const torch::TensorAccessor<scalar_t, 1> input,
                             ParticleData<scalar_t> &tree_data,
                             const int current_max_level) {

    auto apr_it = apr.iterator();
    auto tree_it = apr.tree_iterator();
    auto parent_it = apr.tree_iterator();

    tree_data.data.resize(tree_it.total_number_particles(current_max_level-1), 0);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared) firstprivate(apr_it, tree_it, parent_it) if(parallel)
#endif
    {

        // reduce downsampled particles onto tree nodes
        if(current_max_level < apr_it.level_max()) {
            const auto offset = compute_tree_offset(apr_it, tree_it, current_max_level);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic, 2)
#endif
            for(int z = 0; z < tree_it.z_num(current_max_level); ++z) {
                for(int x = 0; x < tree_it.x_num(current_max_level); ++x) {

                    const float scale_factor_xz = _compute_scale_factor_xz(tree_it, parent_it, current_max_level, z, x);
                    const float scale_factor_yxz = _compute_scale_factor_yxz(scale_factor_xz, tree_it, parent_it, current_max_level);

                    parent_it.begin(current_max_level - 1, z / 2, x / 2);

                    for (tree_it.begin(current_max_level, z, x); tree_it < tree_it.end(); ++tree_it) {

                        while (parent_it.y() < (tree_it.y() / 2)) { parent_it++; }

                        const auto scale_factor = (parent_it.y() == (parent_it.y_num(current_max_level - 1) - 1))
                                                  ? scale_factor_yxz : scale_factor_xz;

                        tree_data[parent_it] += scale_factor * input[tree_it + offset];
                    } // y
                } // x
            } // z
        } // if current_level_max

        // reduce apr particles onto tree nodes
        for (int level = current_max_level; level >= apr_it.level_min(); --level) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic, 2)        // chunksize 2 to avoid race conditions
#endif
            for (int z = 0; z < apr_it.z_num(level); ++z) {
                for (int x = 0; x < apr_it.x_num(level); ++x) {

                    const float scale_factor_xz = _compute_scale_factor_xz(apr_it, parent_it, level, z, x);
                    const float scale_factor_yxz = _compute_scale_factor_yxz(scale_factor_xz, apr_it, parent_it, level);

                    parent_it.begin(level - 1, z / 2, x / 2);

                    for (apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        while (parent_it.y() < (apr_it.y() / 2)) { parent_it++; }

                        const float scale_factor = (parent_it.y() == (parent_it.y_num(level - 1) - 1))
                                                   ? scale_factor_yxz : scale_factor_xz;

                        tree_data[parent_it] += scale_factor * input[apr_it];
                    } // y
                } // x
            } // z
        } // level

        // then do the rest of the tree where order matters
        for (int level = std::min(current_max_level-1, tree_it.level_max()); level > tree_it.level_min(); --level) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic, 2)        // chunksize 2 to avoid race conditions
#endif
            for (int z = 0; z < tree_it.z_num(level); ++z) {
                for (int x = 0; x < tree_it.x_num(level); ++x) {

                    const float scale_factor_xz = _compute_scale_factor_xz(tree_it, parent_it, level, z, x);
                    const float scale_factor_yxz = _compute_scale_factor_yxz(scale_factor_xz, tree_it, parent_it, level);

                    parent_it.begin(level - 1, z / 2, x / 2);

                    for (tree_it.begin(level, z, x); tree_it < tree_it.end(); ++tree_it) {

                        while (parent_it.y() < (tree_it.y() / 2)) { parent_it++; }

                        const auto scale_factor = (parent_it.y() == (parent_it.y_num(level - 1) - 1)) ? scale_factor_yxz
                                                                                                      : scale_factor_xz;

                        tree_data[parent_it] += scale_factor * tree_data[tree_it];
                    } // y
                } // x
            } // z
        } // level
    } // parallel region
} // fill_tree_mean


template<bool parallel, typename scalar_t>
void helpers::fill_tree_mean_backward(APR &apr,
                                      torch::TensorAccessor<scalar_t, 1> grad_input,
                                      ParticleData<scalar_t> &grad_tree,
                                      const int current_max_level) {

    auto apr_it = apr.iterator();
    auto tree_it = apr.tree_iterator();
    auto parent_it = apr.tree_iterator();

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel default(shared) firstprivate(apr_it, tree_it, parent_it) if(parallel)
#endif
    { //parallel region

        // push tree gradients from root to leaf
        for (int level = tree_it.level_min()+1; level <= std::min(current_max_level-1, tree_it.level_max()); ++level) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic)
#endif
            for (int z = 0; z < tree_it.z_num(level); ++z) {
                for (int x = 0; x < tree_it.x_num(level); ++x) {

                    const float scale_factor_xz = _compute_scale_factor_xz(tree_it, parent_it, level, z, x);
                    const float scale_factor_yxz = _compute_scale_factor_yxz(scale_factor_xz, tree_it, parent_it, level);

                    parent_it.begin(level - 1, z / 2, x / 2);

                    for (tree_it.begin(level, z, x); tree_it < tree_it.end(); ++tree_it) {

                        while (parent_it.y() < (tree_it.y() / 2)) { parent_it++; }

                        const auto scale_factor = (parent_it.y() == (parent_it.y_num(level - 1) - 1)) ? scale_factor_yxz
                                                                                                      : scale_factor_xz;

                        grad_tree[tree_it] += scale_factor * grad_tree[parent_it];
                    } // y
                } // x
            } // z
        } // level

        // push accumulated tree gradients to APR particle gradients
        for (int level = current_max_level; level >= apr_it.level_min(); --level) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic)
#endif
            for (int z = 0; z < apr_it.z_num(level); ++z) {
                for (int x = 0; x < apr_it.x_num(level); ++x) {

                    //dealing with boundaries
                    const float scale_factor_xz = _compute_scale_factor_xz(apr_it, parent_it, level, z, x);
                    const float scale_factor_yxz = _compute_scale_factor_yxz(scale_factor_xz, apr_it, parent_it, level);

                    parent_it.begin(level - 1, z / 2, x / 2);

                    for (apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        while (parent_it.y() < (apr_it.y() / 2)) { parent_it++; }

                        const float scale_factor = (parent_it.y() == (parent_it.y_num(level - 1) - 1))
                                                   ? scale_factor_yxz : scale_factor_xz;

                        grad_input[apr_it] += scale_factor * grad_tree[parent_it];
                    } // y
                } // x
            } // z
        } // level

        // push accumulated tree gradients to downsampled particle gradients
        if(current_max_level < apr_it.level_max()) {
            auto offset = compute_tree_offset(apr_it, tree_it, current_max_level);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp for schedule(dynamic)
#endif
            for(int z = 0; z < tree_it.z_num(current_max_level); ++z) {
                for(int x = 0; x < tree_it.x_num(current_max_level); ++x) {

                    const float scale_factor_xz = _compute_scale_factor_xz(tree_it, parent_it, current_max_level, z, x);
                    const float scale_factor_yxz = _compute_scale_factor_yxz(scale_factor_xz, tree_it, parent_it, current_max_level);

                    parent_it.begin(current_max_level - 1, z / 2, x / 2);

                    for (tree_it.begin(current_max_level, z, x); tree_it < tree_it.end(); ++tree_it) {

                        while (parent_it.y() < (tree_it.y() / 2)) { parent_it++; }

                        const auto scale_factor = (parent_it.y() == (parent_it.y_num(current_max_level - 1) - 1))
                                                  ? scale_factor_yxz : scale_factor_xz;

                        grad_input[tree_it + offset] += scale_factor * grad_tree[parent_it];
                    } // y
                } // x
            } // z
        } // if current_level_max
    } // parallel region
} // fill_tree_mean


constexpr int helpers::compute_number_parent_levels(const int sz, const int sx, const int sy) {
    return std::ceil(std::log2((std::max(std::max(sz, sx), sy) - 1)/2 + 2) - 1);
}


template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
void helpers::update_dense_array(LinearIterator &apr_it,
                                 LinearIterator &tree_it,
                                 const int level,
                                 const int z,
                                 const int x,
                                 const torch::TensorAccessor<scalar_t, 1> input,
                                 const ParticleData<scalar_t> &tree_data,
                                 APRNetImageBuffer<scalar_t> &patch_buffer,
                                 const int current_max_level) {

    const uint64_t mesh_offset = ((z + (stencilSizeZ-1)/2) % stencilSizeZ) * patch_buffer.x_num * patch_buffer.y_num +
                                 ((x + (stencilSizeX-1)/2) % stencilSizeX) * patch_buffer.y_num +
                                 (stencilSizeY-1)/2;

    // fill in same level values
    for (apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
        patch_buffer.mesh[apr_it.y() + mesh_offset] = input[apr_it];
    }

    // fill in coarser APR particles
    if(level > apr_it.level_min()) {
        const int num_parent_levels = std::min(compute_number_parent_levels(stencilSizeZ, stencilSizeX, stencilSizeY),
                                               level - apr_it.level_min());

        for (int dlevel = 1; dlevel <= num_parent_levels; ++dlevel) {
            const int step_size = std::pow(2, dlevel);
            for (apr_it.begin(level - dlevel, z / step_size, x / step_size); apr_it < apr_it.end(); ++apr_it) {
                const int y = step_size * apr_it.y();
                const int y_m = std::min(y + step_size, apr_it.y_num(level));

                for(int y_p = y; y_p < y_m; ++y_p) {
                    patch_buffer.mesh[mesh_offset + y_p] = input[apr_it];
                }
            }
        }
    }

    // fill in finer APR particle values via the APR tree
    if(level < apr_it.level_max()) {
        if(level == current_max_level) {
            const auto tree_offset = compute_tree_offset(apr_it, tree_it, level);
            for (tree_it.begin(level, z, x); tree_it < tree_it.end(); ++tree_it) {
                patch_buffer.mesh[tree_it.y() + mesh_offset] = input[tree_it + tree_offset];
            }
        } else {
            for (tree_it.begin(level, z, x); tree_it < tree_it.end(); ++tree_it) {
                patch_buffer.mesh[tree_it.y() + mesh_offset] = tree_data[tree_it];
            }
        }
    }

    // zero pad the boundary of the buffer
    const uint64_t base_offset = mesh_offset - (stencilSizeY-1)/2;

    for(int y = 0; y < (stencilSizeY-1)/2; ++y) {
        patch_buffer.mesh[base_offset + y] = 0;
    }

    for(int y = 0; y < (stencilSizeY-1)/2; ++y) {
        patch_buffer.mesh[base_offset + patch_buffer.y_num - 1 - y] = 0;
    }
}


template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
void helpers::update_dense_array_backward(LinearIterator &apr_it,
                                          LinearIterator &tree_it,
                                          const int level,
                                          const int z,
                                          const int x,
                                          std::vector<scalar_t> &grad_input,
                                          ParticleData<scalar_t> &grad_tree,
                                          APRNetImageBuffer<scalar_t> &grad_patch,
                                          const int current_max_level) {

    const uint64_t mesh_offset = ((z + (stencilSizeZ-1)/2) % stencilSizeZ) * grad_patch.x_num * grad_patch.y_num +
                                 ((x + (stencilSizeX-1)/2) % stencilSizeX) * grad_patch.y_num +
                                 (stencilSizeY-1)/2;

    // fill in same level values
    for (apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
        grad_input[apr_it] += grad_patch.mesh[apr_it.y() + mesh_offset];
    }

    // fill in coarser APR particles
    if(level > apr_it.level_min()) {
        const int num_parent_levels = std::min(compute_number_parent_levels(stencilSizeZ, stencilSizeX, stencilSizeY),
                                               level - apr_it.level_min());

        for (int dlevel = 1; dlevel <= num_parent_levels; ++dlevel) {
            const int step_size = std::pow(2, dlevel);
            for (apr_it.begin(level - dlevel, z / step_size, x / step_size); apr_it < apr_it.end(); ++apr_it) {
                const int y = step_size * apr_it.y();
                const int y_m = std::min(y + step_size, apr_it.y_num(level));

                for(int y_p = y; y_p < y_m; ++y_p) {
                    grad_input[apr_it] += grad_patch.mesh[mesh_offset + y_p];
                }
            }
        }
    }

    // fill in finer APR particle values via the APR tree
    if(level < apr_it.level_max()) {
        if(level == current_max_level) {
            const auto tree_offset = compute_tree_offset(apr_it, tree_it, current_max_level);
            for (tree_it.begin(current_max_level, z, x); tree_it < tree_it.end(); ++tree_it) {
                grad_input[tree_it + tree_offset] += grad_patch.mesh[tree_it.y() + mesh_offset];
            }
        } else {
            for (tree_it.begin(level, z, x); tree_it < tree_it.end(); ++tree_it) {
                grad_tree[tree_it] += grad_patch.mesh[tree_it.y() + mesh_offset];
            }
        }
    }

    // zero row in grad_patch
    auto begin = grad_patch.mesh.begin() + mesh_offset;
    std::fill(begin, begin + apr_it.y_num(level), scalar_t(0));
}


template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
void helpers::zero_boundary_x(const int x, APRNetImageBuffer<scalar_t> &patch_buffer) {
    for(int z = 0; z < stencilSizeZ; ++z) {
        const uint64_t out_offset = patch_buffer.offset(z, x % stencilSizeX);

        std::fill(patch_buffer.mesh.begin() + out_offset,
                  patch_buffer.mesh.begin() + out_offset + patch_buffer.y_num,
                  scalar_t(0));
    }
}


template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
void helpers::zero_boundary_z(const int z, APRNetImageBuffer<scalar_t> &patch_buffer) {

    const uint64_t out_offset = patch_buffer.offset_z(z % stencilSizeZ);
    const uint64_t slice_size = uint64_t(patch_buffer.x_num) * patch_buffer.y_num;

    std::fill(patch_buffer.mesh.begin() + out_offset,
              patch_buffer.mesh.begin() + out_offset + slice_size,
              scalar_t(0));
}


template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
void helpers::accumulate_convolution(LinearIterator& apr_it,
                                     LinearIterator& tree_it,
                                     const int level,
                                     const int z,
                                     const int x,
                                     const APRNetImageBuffer<scalar_t>& patch,
                                     const torch::TensorAccessor<scalar_t, 4> filter,
                                     torch::TensorAccessor<scalar_t, 2> output,
                                     std::vector<scalar_t>& temp_vec,
                                     const int current_max_level) {

    const int y_num = patch.y_num;
    const int xy_num = patch.x_num * y_num;

    // iterate over input row
    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
        const int y = apr_it.y();
        int counter = 0;

        // copy input patch around current particle to temp_vec
        for (int sz = 0; sz < stencilSizeZ; ++sz) {
            uint64_t base_offset = ((z + sz) % stencilSizeZ) * xy_num + y;
            for (int sx = 0; sx < stencilSizeX; ++sx) {
                uint64_t offset = base_offset + ((x + sx) % stencilSizeX) * y_num;
                for (int sy = 0; sy < stencilSizeY; ++sy) {
                    temp_vec[counter++] = patch.mesh[offset+sy];
                }
            }
        }

        // accumulate convolution result for each output channel
        for (int ch_out = 0; ch_out < filter.size(0); ++ch_out) {
            const scalar_t* w_ptr = filter[ch_out].data();
            scalar_t res = 0;
            for(int i = 0; i < stencilSizeZ*stencilSizeX*stencilSizeY; ++i) {
                res += temp_vec[i] * w_ptr[i];
            }
            output[ch_out][apr_it] += res;
        }
    }

    if(level < apr_it.level_max() && level == current_max_level) {
        const auto tree_offset = compute_tree_offset(apr_it, tree_it, level);
        for (tree_it.begin(level, z, x); tree_it < tree_it.end(); ++tree_it) {
            const int y = tree_it.y();
            int counter = 0;

            // copy input patch around current particle to temp_vec
            for (int sz = 0; sz < stencilSizeZ; ++sz) {
                uint64_t base_offset = ((z + sz) % stencilSizeZ) * xy_num + y;
                for (int sx = 0; sx < stencilSizeX; ++sx) {
                    uint64_t offset = base_offset + ((x + sx) % stencilSizeX) * y_num;
                    for (int sy = 0; sy < stencilSizeY; ++sy) {
                        temp_vec[counter++] = patch.mesh[offset + sy];
                    }
                }
            }

            // accumulate convolution result for each output channel
            for (int ch_out = 0; ch_out < filter.size(0); ++ch_out) {
                const scalar_t *w_ptr = filter[ch_out].data();
                scalar_t res = 0;
                for (int i = 0; i < stencilSizeZ * stencilSizeX * stencilSizeY; ++i) {
                    res += temp_vec[i] * w_ptr[i];
                }
                output[ch_out][tree_it + tree_offset] += res;
            }
        }
    }
}


template<typename scalar_t, int stencilSizeZ, int stencilSizeX, int stencilSizeY>
void helpers::accumulate_convolution_backward(LinearIterator &apr_it,
                                              LinearIterator &tree_it,
                                              const int level,
                                              const int z,
                                              const int x,
                                              const APRNetImageBuffer<scalar_t> &patch,
                                              APRNetImageBuffer<scalar_t> &grad_patch,
                                              const torch::TensorAccessor<scalar_t, 4> filter,
                                              const torch::TensorAccessor<scalar_t, 2> grad_output,
                                              std::vector<scalar_t> &temp_dw,
                                              std::vector<scalar_t> &temp_vec,
                                              const int current_max_level,
                                              const size_t w_offset) {

    const size_t y_num = patch.y_num;
    const size_t xy_num = patch.x_num * y_num;
    const int stencil_size = stencilSizeZ*stencilSizeX*stencilSizeY;

    /// accumulate input gradients in grad_patch
    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
        const int y = apr_it.y();
        std::fill(temp_vec.begin(), temp_vec.end(), scalar_t(0));

        // accumulate gradient for each output channel
        for (int ch_out = 0; ch_out < filter.size(0); ++ch_out) {
            const scalar_t dO = grad_output[ch_out][apr_it];
            const scalar_t* w_ptr = filter[ch_out].data();
            for(int i = 0; i < stencil_size; ++i) {
                temp_vec[i] += dO * w_ptr[i];
            }
        }

        // add to grad_patch
        int counter = 0;
        for (int sz = 0; sz < stencilSizeZ; ++sz) {
            uint64_t base_offset = ((z + sz) % stencilSizeZ) * xy_num + y;
            for (int sx = 0; sx < stencilSizeX; ++sx) {
                uint64_t offset = base_offset + ((x + sx) % stencilSizeX) * y_num;
                for (int sy = 0; sy < stencilSizeY; ++sy) {
                    grad_patch.mesh[offset + sy] += temp_vec[counter++];
                }
            }
        }
    }

    /// accumulate weight gradients in grad_filter
    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
        const int y = apr_it.y();
        int counter = 0;

        // copy input patch around current particle to temp_vec
        for (int sz = 0; sz < stencilSizeZ; ++sz) {
            const uint64_t base_offset = ((z + sz) % stencilSizeZ) * xy_num + y;
            for (int sx = 0; sx < stencilSizeX; ++sx) {
                const uint64_t offset = base_offset + ((x + sx) % stencilSizeX) * y_num;
                for (int sy = 0; sy < stencilSizeY; ++sy) {
                    temp_vec[counter++] = patch.mesh[offset+sy];
                }
            }
        }

        // accumulate weight gradient
        for (int ch_out = 0; ch_out < filter.size(0); ++ch_out) {
            const scalar_t dO = grad_output[ch_out][apr_it];
            const size_t offset = w_offset + stencil_size * ch_out;
            for(int i = 0; i < stencil_size; ++i) {
                temp_dw[i + offset] += dO * temp_vec[i];
            }
        }
    }

    /// do the same for downsampled particle locations
    if(level < apr_it.level_max() && level == current_max_level) {
        const auto tree_offset = compute_tree_offset(apr_it, tree_it, current_max_level);

        /// accumulate input gradients in grad_patch
        for(tree_it.begin(level, z, x); tree_it < tree_it.end(); ++tree_it) {
            const int y = tree_it.y();
            std::fill(temp_vec.begin(), temp_vec.end(), scalar_t(0));

            for (int ch_out = 0; ch_out < filter.size(0); ++ch_out) {
                const scalar_t dO = grad_output[ch_out][tree_it + tree_offset];
                const scalar_t* w_ptr = filter[ch_out].data();
                for(int i = 0; i < stencil_size; ++i) {
                    temp_vec[i] += dO * w_ptr[i];
                }
            }

            int counter = 0;
            for (int sz = 0; sz < stencilSizeZ; ++sz) {
                size_t base_offset = ((z + sz) % stencilSizeZ) * xy_num + y;
                for (int sx = 0; sx < stencilSizeX; ++sx) {
                    size_t offset = base_offset + ((x + sx) % stencilSizeX) * y_num;
                    for (int sy = 0; sy < stencilSizeY; ++sy) {
                        grad_patch.mesh[offset+sy] += temp_vec[counter++];
                    }
                }
            }
        }


        /// accumulate weight gradients in grad_filter
        for(tree_it.begin(level, z, x); tree_it < tree_it.end(); ++tree_it) {
            const int y = tree_it.y();
            int counter = 0;

            for (int sz = 0; sz < stencilSizeZ; ++sz) {
                size_t base_offset = ((z + sz) % stencilSizeZ) * xy_num + y;
                for (int sx = 0; sx < stencilSizeX; ++sx) {
                    size_t offset = base_offset + ((x + sx) % stencilSizeX) * y_num;
                    for (int sy = 0; sy < stencilSizeY; ++sy) {
                        temp_vec[counter++] = patch.mesh[offset+sy];
                    }
                }
            }

            for (int ch_out = 0; ch_out < filter.size(0); ++ch_out) {
                const scalar_t dO = grad_output[ch_out][tree_it + tree_offset];
                const size_t offset = w_offset + stencil_size * ch_out;
                for(int i = 0; i < stencil_size; ++i) {
                    temp_dw[i + offset] += dO * temp_vec[i];
                }
            }
        }
    }
}


#endif //APRNET_HELPERS_HPP
