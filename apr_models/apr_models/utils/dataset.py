from os.path import join, basename, exists
import pyapr
from pyapr.aprnet import stardist_utils
from torch.utils.data import Dataset, Subset
import torch
import numpy as np
from glob import glob
from typing import Union, List, Any, Optional, Callable, Tuple
from skimage import io as skio


class APRDataset(Dataset):
    def __init__(self,
                 root_dir: Union[str, List[str]],
                 parts_name: str = 'particles',
                 labels_name: Union[str, Any] = None,
                 transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.file_list = glob(join(root_dir, '*.apr')) if isinstance(root_dir, str) else [y for x in root_dir for y in glob(join(x, '*.apr'))]
        self.parts_name = parts_name
        self.labels_name = labels_name
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        path = self.file_list[index]
        apr, parts = pyapr.io.read(path, parts_name=self.parts_name)
        if self.labels_name:
            labs = pyapr.io.read_particles(path, parts_name=self.labels_name)
            if self.transform:
                parts, labs = self.transform(parts, labs)
            return apr, parts, labs

        if self.transform:
            parts, _ = self.transform(parts)
        return apr, parts, None


class StarDistDataset(Dataset):
    def __init__(self,
                 rays,
                 data_dir: Union[str, List[str]],
                 label_dir: Optional[Union[str, List[str]]] = None,
                 label_tag: str = '',
                 parts_name: str = 'particles',
                 pixels: bool = False,
                 file_ext: Optional[str] = None,
                 crop_size: Optional[Tuple[int, int, int]] = None,
                 rel_error: Union[float, Tuple[float, float]] = 0.1,
                 transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.pixels = pixels
        self.rel_error = rel_error
        self.crop_size = crop_size
        file_ext = file_ext or '*.tif' if pixels else '*.apr'
        self.file_list = glob(join(data_dir, file_ext)) if isinstance(data_dir, str) else [y for x in data_dir for y in glob(join(x, file_ext))]
       
        self.labels = label_dir is not None and exists(label_dir)
        if not self.labels:
            print('labels directory {label_dir} does not exist - StarDistDataset will not yield ground truth data')
        self.rays = rays
        self.parts_name = parts_name
        self.label_tag = label_tag
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data_path = self.file_list[index]
        if self.pixels:
            img = skio.imread(data_path)
            par = pyapr.APRParameters()
            par.rel_error = self.rel_error if isinstance(self.rel_error, float) else np.random.uniform(low=self.rel_error[0], high=self.rel_error[1])
            par.auto_parameters = True
            if self.crop_size is not None:
                assert len(self.crop_size) == 3
                sz = [min(self.crop_size[i], img.shape[i]) for i in range(3)]
                high = [max(img.shape[i] - sz[i], 1) for i in range(3)]
                corner = np.random.randint(0, high)
                img = np.ascontiguousarray(img[corner[0]:corner[0]+sz[0], corner[1]:corner[1]+sz[1], corner[2]:corner[2]+sz[2]])
            apr, parts = pyapr.converter.get_apr(img, params=par, verbose=False)
        else:
            apr, parts = pyapr.io.read(data_path, parts_name=self.parts_name)
        
        dist = None
        prob = None

        if self.labels:
            fname = basename(data_path)[:-4]
            label_path = join(self.label_dir, f'{fname}{self.label_tag}.tif')
            label_image = skio.imread(label_path)
            if self.pixels and self.crop_size:
                label_image = np.ascontiguousarray(label_image[corner[0]:corner[0]+sz[0], corner[1]:corner[1]+sz[1], corner[2]:corner[2]+sz[2]])
            dist = stardist_utils.labels_to_dist(apr, label_image, self.rays)
            prob = stardist_utils.get_prob_apr(apr, label_image)

        if self.transform:
            parts = self.transform(parts)

        return apr, parts, dist, prob


def split_dataset(dataset: APRDataset, sizes: Union[int, List[int]], transforms: Optional[Union[Callable, List[Callable]]] = None):
    """
    Splits an APRDataset into random subsets of size `[*sizes, len(dataset)-sum(sizes)]`

    Parameters
    ----------
    dataset: APRDataset
        The dataset to split
    sizes: int or list of ints
        Size(s) of the subsets to extract
    transforms: Callable or list of Callable (optional)
        Transforms to be used in the __getitem__ method of each subset. Note: does not override the transform of the
        input dataset.
    """
    if isinstance(sizes, int):
        sizes = [sizes]

    n_subsets = len(sizes) + 1
    if isinstance(transforms, list):
        assert len(transforms) == n_subsets, 'Number of provided transforms (%d) does not match number of subsets (%d)' % (len(transforms), n_subsets)
        tfs = transforms
    else:
        tfs = [transforms] * n_subsets

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    splits = [0] + list(np.cumsum(sizes)) + [len(dataset)]
    subsets = [Subset(SplitDataset(dataset, transform=tfs[i]), indices[splits[i]:splits[i+1]]) for i in range(n_subsets)]
    return tuple(subsets)


class collate_intensity_and_labels():
    def __init__(self, labels_drop_channel_dim=False, label_dtype=torch.float32, input_levels=False):
        self.labels_drop_channel_dim = labels_drop_channel_dim
        self.label_dtype = label_dtype
        self.input_levels = input_levels

    def __call__(self, data):
        x, y, z = zip(*data)
        aprs = list(x)
        npartmax = max([a.total_number_particles() for a in aprs])
        intensities = torch.zeros((len(aprs), 2 if self.input_levels else 1, npartmax), dtype=torch.float)
        labels_shape = (len(aprs), npartmax) if self.labels_drop_channel_dim else (len(aprs), 1, npartmax)
        labels = torch.zeros(labels_shape, dtype=self.label_dtype)
        
        for i, (p, l) in enumerate(zip(y, z)):
            intensities[i, 0, :len(p)] = torch.from_numpy(np.array(p))
            if self.input_levels:
                tmp = pyapr.FloatParticles()
                tmp.fill_with_levels(aprs[i])
                tmp = tmp * (1/aprs[i].level_max())
                intensities[i, 1, :len(tmp)] = torch.from_numpy(np.array(tmp))
            labels[i, ..., :len(l)] = torch.from_numpy(np.array(l).astype(np.float32))
        return aprs, intensities, labels


class collate_intensity:
    def __call__(self, data):
        x, y, _ = zip(*data)
        aprs = list(x)
        npartmax = max([a.total_number_particles() for a in aprs])
        intensities = torch.zeros((len(aprs), 1, npartmax), dtype=torch.float)
        for i, p in enumerate(y):
            intensities[i, 0, :len(p)] = torch.from_numpy(np.array(p))
        return aprs, intensities


class collate_stardist:
    def __call__(self, data):
        _apr, _parts, _dist, _prob = zip(*data)
        aprs = list(_apr)
        npartmax = max([a.total_number_particles() for a in aprs])
        intensities = torch.zeros((len(aprs), 1, npartmax), dtype=torch.float)
        for i, parts in enumerate(_parts):
            intensities[i, 0, :len(parts)] = torch.from_numpy(np.array(parts))
        
        if _dist[0] is not None:
            prob = torch.zeros((len(aprs), 1, npartmax), dtype=torch.float)
            dist = torch.zeros((len(aprs), _dist[0].shape[1], npartmax), dtype=torch.float)
            for i, (d, p) in enumerate(zip(_dist, _prob)):
                prob[i, 0, :len(parts)] = torch.from_numpy(p)
                dist[i, :, :len(parts)] = torch.from_numpy(d).transpose(1, 0)
            return aprs, intensities, prob, dist
        else:
            return aprs, intensities, None, None

