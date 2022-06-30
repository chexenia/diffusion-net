import os
import sys
import numpy as np
import h5py
import pickle
import re
import torch
from torch.utils.data import Dataset
from reco2.scan2cad.utils.feature_set import Features
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../src/")
)  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP

from reco2.extensions.SPFN.spfn.fitter_factory import (
    register_primitives,
    SPFN_DEFAULT_PRIMITIVES,
)

register_primitives(SPFN_DEFAULT_PRIMITIVES)


def create_unit_data_from_hdf5(f, n_max_instances=256, noisy=True, shuffle=True):
    """
    f will be a h5py group-like object
    """
    from reco2.extensions.SPFN.spfn import fitter_factory

    P = f["noisy_points"][()] if noisy else f["gt_points"][()]  # Nx3
    normal_gt = f["gt_normals"][()]  # Nx3
    I_gt = f["gt_labels"][()]  # N

    P_gt = []

    # next check if soup_ids are consecutive
    found_soup_ids = []
    soup_id_to_key = {}
    soup_prog = re.compile("(.*)_soup_([0-9]+)$")
    for key in list(f.keys()):
        m = soup_prog.match(key)
        if m is not None:
            soup_id = int(m.group(2))
            found_soup_ids.append(soup_id)
            soup_id_to_key[soup_id] = key
    found_soup_ids.sort()
    n_instances = len(found_soup_ids)
    if n_instances == 0:
        print("zero soup instances")
        return None
    for i in range(n_instances):
        if i not in found_soup_ids:
            print("{} is not found in soup ids!".format(i))
            return None

    instances = []
    for i in range(n_instances):
        g = f[soup_id_to_key[i]]
        P_gt_cur = g["gt_points"][()]
        P_gt.append(P_gt_cur)
        meta = pickle.loads(g.attrs["meta"])
        primitive = fitter_factory.create_primitive_from_dict(meta)
        if primitive is None:
            return None
        instances.append(primitive)

    if n_max_instances != -1 and n_instances > n_max_instances:
        print(
            "n_instances {} > n_max_instances {}".format(n_instances, n_max_instances)
        )
        return None

    if np.amax(I_gt) >= n_instances:
        print("max label {} > n_instances {}".format(np.amax(I_gt), n_instances))
        return None

    T_gt = [
        fitter_factory.primitive_name_to_id(primitive.get_primitive_name())
        for primitive in instances
    ]
    if n_max_instances != -1:
        T_gt.extend([0 for _ in range(n_max_instances - n_instances)])  # K

    n_total_points = P.shape[0]
    n_gt_points_per_instance = P_gt[0].shape[0]
    P_gt.extend(
        [
            np.zeros(dtype=float, shape=[n_gt_points_per_instance, 3])
            for _ in range(n_max_instances - n_instances)
        ]
    )

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
        "P": P,
        "normal_gt": normal_gt,
        "P_gt": P_gt,
        "I_gt": I_gt,
        "T_gt": T_gt,
    }

    # Next put in primitive parameters
    parts = []
    for fitter_cls in fitter_factory.all_fitter_classes:
        parts.append(
            fitter_cls.extract_parameter_data_as_dict(instances, n_max_instances)
        )

    return result, parts


def preprocess_data(fpath, k_eig=128, noisy=True, shuffle=True):
    with h5py.File(fpath, "r") as handle:
        data, _ = create_unit_data_from_hdf5(
            handle, n_max_instances=256, noisy=noisy, shuffle=shuffle
        )
        handle.close()
    verts = data["P"]
    # seg_labels = data['I_gt']
    type_per_segment = data["T_gt"]
    segm_per_point = data["I_gt"]
    type_per_point = [type_per_segment[s] for s in segm_per_point]

    labels = type_per_point
    normals = data["normal_gt"]

    # to torch
    faces = None  # spfn data is point cloud
    verts = torch.tensor(np.ascontiguousarray(verts)).float()
    labels = torch.tensor(np.ascontiguousarray(labels))
    normals = torch.tensor(np.ascontiguousarray(normals)).float()

    # center and unit scale
    verts = diffusion_net.geometry.normalize_positions(verts)

    # Precompute operators
    (
        frames,
        massvec,
        L,
        evals,
        evecs,
        gradX,
        gradY,
    ) = diffusion_net.geometry.compute_operators(verts, faces, k_eig, normals)

    return verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals


class SPFNDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split_path,
        in_features,
        k_eig=128,
        use_cache=True,
    ):
        self.n_max_instances = 256
        self.noisy = True
        self.root_dir = root_dir
        self.split_path = split_path
        with open(split_path, "r") as f:
            self.hdf5_file_list = f.read().splitlines()
            f.close()

        self.hdf5_file_list = [
            f for f in self.hdf5_file_list if (self.root_dir / f).exists()
        ]

        self.use_cache = use_cache
        self.in_features = in_features
        self.k_eig = k_eig
        self.cache_dir = root_dir / f"cache.diffnet.keig{k_eig}"
        self.shuffle = True  # shuffle points in original point cloud

        # Randomly rotate positions
        self.augment_random_rotate = in_features == Features.XYZ

    def __len__(self):
        return len(self.hdf5_file_list)

    def get_relpath(self, i):
        return self.hdf5_file_list[i]

    def get_cache_path(self, idx):
        return (self.cache_dir / f"{self.hdf5_file_list[idx]}").with_suffix(".pt")

    def retrieve_cache(self, idx):
        cached_data_path = self.get_cache_path(idx)

        if self.use_cache and cached_data_path.exists():
            (
                verts,
                faces,
                frames,
                massvec,
                L,
                evals,
                evecs,
                gradX,
                gradY,
                labels,
                normals,
            ) = torch.load(cached_data_path)
            return verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals
        else:
            return None
    
    def store_cache(self, verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals, idx):
        if self.use_cache:
            cached_data_path = self.get_cache_path(idx)
            if not cached_data_path.exists():
                cached_data_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    (
                        verts,
                        faces,
                        frames,
                        massvec,
                        L,
                        evals,
                        evecs,
                        gradX,
                        gradY,
                        labels,
                        normals,
                    ),
                    cached_data_path,
                )

    def __getitem__(self, idx):
        data = self.retrieve_cache(idx)
        if data is not None:
            verts, faces, frames, massvec, L, evals,
            evecs,
            gradX,
            gradY,
            labels,
            normals = data
        else:
            (
                verts,
                faces,
                frames,
                massvec,
                L,
                evals,
                evecs,
                gradX,
                gradY,
                labels,
                normals,
            ) = preprocess_data(
                self.root_dir / self.hdf5_file_list[idx],
                self.k_eig,
                self.noisy,
                self.shuffle,
            )

            if self.augment_random_rotate: #FIXME: should we be doing it for verts only, what about normals?
                verts = diffusion_net.utils.random_rotate_points(verts)

            # Construct features
            if Features.XYZ == self.in_features:
                features = verts
            elif Features.HKS == self.in_features:
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
            else:
                raise ValueError(f"Unsupported feature type: {self.in_features}")

            # for feat in Features:
            #     if (
            #         feat not in self.in_features
            #         #and feat not in self.out_features
            #         #and feat not in self.extra_features
            #     ):
            #         continue
            #     if feat == Features.XYZ:
            #         features = np.asarray(pcd.points, dtype=np.float32)

            #         if self.augment:
            #             rotation = Rotation.random()
            #             features = rotation.apply(features)
            self.store_cache(verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals, idx)
            
        return (
            verts,
            frames,
            massvec,
            L,
            evals,
            evecs,
            gradX,
            gradY,
            labels,
            normals,
            features,
            idx,
        )

def operate_cache():

    device = verts.device
    dtype = verts.dtype

    if op_cache_dir is not None:
        utils.ensure_dir_exists(op_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                op_cache_dir, hash_key_str + "_" + str(i_cache_search) + ".npz"
            )

            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (
                    not np.array_equal(faces, cache_faces)
                ):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    continue

                # print("  cache hit!")

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache:
                    print("  overwriting cache by request")
                    os.remove(search_path)
                    break

                if cache_k_eig < k_eig:
                    print("  overwriting cache --- not enough eigenvalues")
                    os.remove(search_path)
                    break

                if "L_data" not in npzfile:
                    print("  overwriting cache --- entries are absent")
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                frames = npzfile["frames"]
                mass = npzfile["mass"]
                L = read_sp_mat("L")
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:, :k_eig]
                gradX = read_sp_mat("gradX")
                gradY = read_sp_mat("gradY")

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                gradX = utils.sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = utils.sparse_np_to_torch(gradY).to(device=device, dtype=dtype)

                found = True

                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break

            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

        dtype_np = np.float32

        # Store it in the cache
        if op_cache_dir is not None:

            L_np = utils.sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = utils.sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = utils.sparse_torch_to_np(gradY).astype(dtype_np)

            np.savez(
                search_path,
                verts=verts_np.astype(dtype_np),
                frames=toNP(frames).astype(dtype_np),
                faces=faces_np,
                k_eig=k_eig,
                mass=toNP(mass).astype(dtype_np),
                L_data=L_np.data.astype(dtype_np),
                L_indices=L_np.indices,
                L_indptr=L_np.indptr,
                L_shape=L_np.shape,
                evals=toNP(evals).astype(dtype_np),
                evecs=toNP(evecs).astype(dtype_np),
                gradX_data=gradX_np.data.astype(dtype_np),
                gradX_indices=gradX_np.indices,
                gradX_indptr=gradX_np.indptr,
                gradX_shape=gradX_np.shape,
                gradY_data=gradY_np.data.astype(dtype_np),
                gradY_indices=gradY_np.indices,
                gradY_indptr=gradY_np.indptr,
                gradY_shape=gradY_np.shape,
            )

