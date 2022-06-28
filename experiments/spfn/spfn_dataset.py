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

def create_unit_data_from_hdf5(f, n_max_instances=256, noisy=True, fixed_order=False, check_only=False, shuffle=True, parse_only=False):
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

    if parse_only:
        return P, instances

    if n_max_instances != -1 and n_instances > n_max_instances:
        print('n_instances {} > n_max_instances {}'.format(n_instances, n_max_instances))
        return None

    if np.amax(I_gt) >= n_instances:
        print('max label {} > n_instances {}'.format(np.amax(I_gt), n_instances))
        return None

    if check_only:
        return True

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

class SPFNDataset(Dataset):
    def __init__(self, root_dir, split_path, n_max_instances, noisy, first_n=-1, fixed_order=False, k_eig=128, use_cache=True, op_cache_dir=None):
        self.n_max_instances = n_max_instances
        self.fixed_order = fixed_order
        self.first_n = first_n
        self.noisy = noisy
        self.root_dir = root_dir
        self.split_path = split_path

        with open(split_path, "r") as f:
            self.hdf5_file_list = f.read().splitlines()
            f.close()

        if not fixed_order:
            random.shuffle(self.hdf5_file_list)
        if first_n != -1:
            self.hdf5_file_list = self.hdf5_file_list[:first_n]

        self.k_eig = k_eig 
        self.cache_dir = root_dir / "cache"
        self.op_cache_dir = op_cache_dir

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []
        self.normals_list = []

        # check the cache
        if use_cache:
            load_cache = self.cache_dir / self.split_path.name
            if load_cache.exists():
                print("using dataset cache path: " + str(load_cache))
                if load_cache:
                    print("  --> loading dataset from cache")
                    self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list, self.normals_list, self.hdf5_file_list = torch.load( load_cache)
                    return
                print("  --> dataset not in cache, repopulating")

        for idx, iFile in enumerate(self.hdf5_file_list):
            print(f"loading {iFile}")
            data, parts = self.fetch_data_at_index(idx)

            verts = data['P']
            #seg_labels = data['I_gt']
            type_per_segment = data['T_gt']
            segm_per_point = data['I_gt']
            type_per_point = [type_per_segment[s] for s in segm_per_point]

            labels = type_per_point
            normals = data['normal_gt']

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            labels = torch.tensor(np.ascontiguousarray(labels))
            normals = torch.tensor(np.ascontiguousarray(normals)).float()

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.labels_list.append(labels)
            self.normals_list.append(normals)
            self.faces_list.append(None)

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir, normals=self.normals_list)

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list, self.normals_list, self.hdf5_file_list), load_cache)


    def fetch_data_at_index(self, i):
        fn = self.root_dir / self.hdf5_file_list[i]
        with h5py.File(fn, 'r') as handle:
            #print('loading from', path)
            data = create_unit_data_from_hdf5(handle, self.n_max_instances, noisy=self.noisy, fixed_order=self.fixed_order, shuffle=not self.fixed_order)
            assert data is not None # assume data are all clean

        return data

    
    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx], self.normals_list[idx], self.hdf5_file_list[idx]
