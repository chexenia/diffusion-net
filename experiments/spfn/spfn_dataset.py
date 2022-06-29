import shutil
import os
import sys
import random
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import time
import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP

import random
import os
import h5py
import pickle
import pandas
import re

from reco2.extensions.SPFN.spfn import fitter_factory

def create_unit_data_from_hdf5(f, n_max_instances=256, noisy=True, shuffle=True):
    ''' 
        f will be a h5py group-like object
    '''

    P = f['noisy_points'][()] if noisy else f['gt_points'][()] # Nx3
    normal_gt = f['gt_normals'][()] # Nx3
    I_gt = f['gt_labels'][()] # N

    P_gt = []

    # next check if soup_ids are consecutive
    found_soup_ids = []
    soup_id_to_key = {}
    soup_prog = re.compile('(.*)_soup_([0-9]+)$')
    for key in list(f.keys()):
        m = soup_prog.match(key)
        if m is not None:
            soup_id = int(m.group(2))
            found_soup_ids.append(soup_id)
            soup_id_to_key[soup_id] = key
    found_soup_ids.sort()
    n_instances = len(found_soup_ids)
    if n_instances == 0:
        print('zero soup instances')
        return None
    for i in range(n_instances):
        if i not in found_soup_ids:
            print('{} is not found in soup ids!'.format(i))
            return None

    instances = []
    for i in range(n_instances):
        g = f[soup_id_to_key[i]]
        P_gt_cur = g['gt_points'][()]
        P_gt.append(P_gt_cur)
        meta = pickle.loads(g.attrs['meta'])
        primitive = fitter_factory.create_primitive_from_dict(meta)
        if primitive is None:
            return None
        instances.append(primitive)

    if n_max_instances != -1 and n_instances > n_max_instances:
        print('n_instances {} > n_max_instances {}'.format(n_instances, n_max_instances))
        return None

    if np.amax(I_gt) >= n_instances:
        print('max label {} > n_instances {}'.format(np.amax(I_gt), n_instances))
        return None

    T_gt = [fitter_factory.primitive_name_to_id(primitive.get_primitive_name()) for primitive in instances]
    if n_max_instances != -1:
        T_gt.extend([0 for _ in range(n_max_instances - n_instances)]) # K

    n_total_points = P.shape[0]
    n_gt_points_per_instance = P_gt[0].shape[0]
    P_gt.extend([np.zeros(dtype=float, shape=[n_gt_points_per_instance, 3]) for _ in range(n_max_instances - n_instances)])

    # convert everything to numpy array
    P_gt = np.array(P_gt)
    T_gt = np.array(T_gt)
    
    if shuffle:
        # shuffle per point information around
        perm = np.random.permutation(n_total_points)
        P = P[perm]
        normal_gt = normal_gt[perm]
        I_gt = I_gt[perm]

    result = {
        'P': P,
        'normal_gt': normal_gt,
        'P_gt': P_gt,
        'I_gt': I_gt,
        'T_gt': T_gt,
    }

    # Next put in primitive parameters
    parts = []
    for fitter_cls in fitter_factory.all_fitter_classes:
        parts.append(fitter_cls.extract_parameter_data_as_dict(instances, n_max_instances))

    return result, parts

def preprocess_data(fpath, k_eig=128, noisy=True, shuffle=True):
    print(fpath)
    with h5py.File(fpath, 'r') as handle:
        data, _ = create_unit_data_from_hdf5(handle, n_max_instances=256, noisy=noisy, shuffle=shuffle)
        handle.close()
    verts = data['P']
    #seg_labels = data['I_gt']
    type_per_segment = data['T_gt']
    segm_per_point = data['I_gt']
    type_per_point = [type_per_segment[s] for s in segm_per_point]

    labels = type_per_point
    normals = data['normal_gt']

    # to torch
    faces = None #spfn data is point cloud
    verts = torch.tensor(np.ascontiguousarray(verts)).float()
    labels = torch.tensor(np.ascontiguousarray(labels))
    normals = torch.tensor(np.ascontiguousarray(normals)).float()

    # center and unit scale
    verts = diffusion_net.geometry.normalize_positions(verts)

    # Precompute operators
    frames, massvec, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces, k_eig, None, normals)

    return verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals

class SPFNDataset(Dataset):
    def __init__(self, root_dir, split_path, n_max_instances, noisy, k_eig=128, input_features="xyz", use_cache=True):
        self.n_max_instances = n_max_instances
        self.noisy = noisy
        self.root_dir = root_dir
        self.split_path = split_path
        with open(split_path, "r") as f:
            self.hdf5_file_list = f.read().splitlines()
            f.close()
        
        self.hdf5_file_list = [f for f in self.hdf5_file_list if (self.root_dir / f).exists()]

        self.use_cache = use_cache
        self.input_features = input_features
        self.k_eig = k_eig 
        self.cache_dir = root_dir / f"cache.diffnet.keig{k_eig}"
        self.shuffle = True #shuffle points in original point cloud

    def __len__(self):
        return len(self.hdf5_file_list)

    def get_file_path(self, i):
        return self.root_dir / self.hdf5_file_list[i]

    def __getitem__(self, idx):
        cached_data_path = (self.cache_dir / f"{self.hdf5_file_list[idx]}").with_suffix(".pt")
        print("chached",cached_data_path)
        if self.use_cache and cached_data_path.exists():
            verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals = torch.load(cached_data_path)
        else:
            verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals = preprocess_data(self.root_dir / self.hdf5_file_list[idx], self.k_eig, self.noisy, self.shuffle)
            if self.use_cache:
                cached_data_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals, cached_data_path)
        return verts, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals
