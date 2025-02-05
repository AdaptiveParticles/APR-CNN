from _pyaprwrapper.data_containers import FloatParticles
from _pyaprwrapper.aprnet.utils import labels_to_dist_cpp, get_particle_coords, sample_image, number_parts
import numpy as np
from stardist.utils import edt_prob


def labels_to_dist(apr, img, rays, step_factor=(1, 1, 1), level_delta=0):
    dst_shape = (number_parts(apr, level_delta), len(rays))
    dist = np.zeros(dst_shape, dtype=np.float32)
    dzs, dxs, dys = rays.vertices.T

    assert img.shape == apr.shape()
    assert img.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    assert dzs.dtype == np.float32
    assert dxs.dtype == np.float32
    assert dys.dtype == np.float32

    labels_to_dist_cpp(apr, img, dzs/step_factor[0], dxs/step_factor[1], dys/step_factor[2], dist, level_delta)
    return dist


def get_prob_apr(apr, img, level_delta=0):
    pix_prob = edt_prob(img)
    parts = FloatParticles()
    sample_image(apr, parts, pix_prob, level_delta)
    return np.array(parts)


def get_particle_coordinates(apr, level_delta=0):
    coords = np.zeros((number_parts(apr, level_delta), 3), dtype=np.float32)
    get_particle_coords(apr, coords, level_delta)
    return coords

