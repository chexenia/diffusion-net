from functools import partial
import glob
from multiprocessing import Pool
import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
import h5py
from tqdm import tqdm
from pathlib import Path
import trimesh
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from spfn_dataset import SPFNDataset, preprocess_data

from reco2.extensions.SPFN.spfn.fitter_factory import register_primitives, SPFN_DEFAULT_PRIMITIVES
register_primitives(SPFN_DEFAULT_PRIMITIVES)


# === Options
base_path = os.path.dirname(__file__)

# Parse a few args

def train_epoch(epoch):
    train_loader = split_loaders['train']
    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 


    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    correct = 0
    total_num = 0
    for data in tqdm(train_loader):

        # Get data
        verts, frames, mass, L, evals, evecs, gradX, gradY, labels, normals = data
        faces = None
        # Move to device
        verts = verts.to(device)
        if faces is not None:
            faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        #normals = normals.to(device)

        # Randomly rotate positions
        if augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # Evaluate loss
        print(preds.shape, labels.shape)
        loss = torch.nn.functional.nll_loss(preds, labels)
        loss.backward()
        
        # track accuracy
        pred_labels = torch.max(preds, dim=1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        this_num = labels.shape[0]
        correct += this_correct
        total_num += this_num

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    return train_acc, loss


# Do an evaluation pass on the test dataset 
def evaluate(loader, save_pred=False):
    
    model.eval()

    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for data in tqdm(loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, normals = data
            #mpath = split_datasets['test'].get
            # Move to device
            verts = verts.to(device)
            if faces is not None:
                faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            loss = torch.nn.functional.nll_loss(preds, labels)

            # track accuracy
            pred_labels = torch.max(preds, dim=1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            this_num = labels.shape[0]
            correct += this_correct
            total_num += this_num

            if save_pred:
                # track accuracy
                pred_labels = torch.max(preds, dim=1).indices.cpu().numpy()

                np.savetxt(Path(mpath).with_suffix(f'.diffnet.{input_features}.txt'), pred_labels, fmt='%d')

    test_acc = correct / total_num
    return test_acc, loss


def train():
    print("Training...")
    
    save_path.mkdir(parents=True, exist_ok=True)

    start = time.time()
    pbar = tqdm(range(n_epoch), desc="Training")
    best_acc = 0
    for epoch in pbar:
        e_start = time.time()
        train_acc, train_loss = train_epoch(epoch)
        test_acc, test_loss = evaluate(split_loaders['test'], save_pred=False)
        pbar.write(f"Epoch {epoch+1}/{n_epoch} done in {time.time()-e_start}s. \n Train acc: {train_acc:0.3f} loss: {train_loss:0.3f} Test acc: {test_acc:0.3f} loss: {test_loss:0.3f}")
        writers["train"].add_scalar(f"loss", train_loss, epoch)
        writers["train"].add_scalar(f"acc", train_acc, epoch)
        writers["test"].add_scalar(f"loss", test_loss, epoch)
        writers["test"].add_scalar(f"acc", test_acc, epoch)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), (save_path / model_name).with_suffix(f"best{epoch}.pth"))

    print(f"Finished in {time.time()-start:.2f}s.")
    print(f" ==> saving last model to {save_path}")
    torch.save(model.state_dict(), save_path / model_name.with_suffix(".last.pth"))

def view():
    print("Viewing...")

    meshes = glob.glob(args.dataset_path + '/**/*.ply', recursive=True)
    
    def color_segments(label, n_class=8):
        from matplotlib import pyplot

        ref_colors = np.array(pyplot.get_cmap("tab20").colors)
        idx = int(label * (ref_colors.shape[0] // n_class))
        color = ref_colors[idx]

        return color

    matrix = np.eye(4)
    matrix[:3, :3] *= 0.1

    for f in meshes:
        mesh = trimesh.load(f)
        mesh.apply_transform(matrix)
        mesh.show()
        labels = np.loadtxt(Path(f).with_suffix(f'.diffnet.{input_features}.txt'))
        face_colors = [color_segments(l) for l in labels]
        pred_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, mesh.face_normals, mesh.vertex_normals, face_colors)
        pred_mesh.show()

def prep_one(cache_dir, dataset_path, fn):
        print(f"loading {fn}")
        fpath = dataset_path / fn

        verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals = preprocess_data(fpath, k_eig, noisy, shuffle)

        torch.save((verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, normals), (cache_dir / fn).with_suffix(".pt"))

def prep(args):
    """"precalculate and store the operators for each model and store in disk cache"""
    cache_dir = args.dataset_path / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_path = args.split_path / "{}.txt".format(split)
        with open(split_path, "r") as f:
            hdf5_file_list = f.read().splitlines()
        f.close()
        
        hdf5_file_list = [f for f in hdf5_file_list if (args.dataset_path / f).exists()]
        print(f"Preprocessing {split} {len(hdf5_file_list)} files")
        start = time.time()

        # per_one_prep = partial(prep_one, cache_dir, args.dataset_path)
        # pool = Pool(processes = 12)
        # pool.map(per_one_prep, hdf5_file_list)

        for fn in hdf5_file_list:
            prep_one(cache_dir, args.dataset_path, fn)

        print("Time elapsed: ", time.time() - start)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test", "eval", "view", "prep"])
    parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'xyz')
    parser.add_argument("--split_path", type=Path, required=True, help="path to the split files")
    parser.add_argument("--dataset_path", type=Path, required=True, help="path to the dataset")
    parser.add_argument("--save_path", type=Path, required=True, help="path to save the results")
    parser.add_argument("--checkpoint_path", type=Path, help="path to the pretrained model")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()

    # system things
    device = torch.device('cuda:0')
    dtype = torch.float32

    #TODO: move to configs
    # problem/dataset things
    n_class = 4

    # model 
    input_features = args.input_features
    k_eig = 128
    n_block = 4

    # training settings
    n_epoch = 200
    lr = 1e-3
    decay_every = 50
    decay_rate = 0.5
    num_workers = 0
    batch_size = 32
    augment_random_rotate = (input_features == 'xyz')
    n_max_instances = 256 #max number of segments for a model
    dataset_name = "spfn"
    output_as = "vertices"
    noisy = True
    shuffle = True
    dataset_path = args.dataset_path
    model_name = f"diffnet.{dataset_name}.seg.{input_features}_{n_block}x{k_eig}"
    cache_name = f"cache.diffnet.keig{k_eig}"
    save_path = args.save_path / model_name

    splits = ["train", "test", "eval"]

    if args.command == "prep":
        prep(args)
    elif args.command == "view":
        view()
    else:

        # === Load datasets

        split_datasets = {}
        split_loaders = {}
        split_samplers = {}
        writers = {}
        for split in splits:
            split_path = args.split_path / "{}.txt".format(split)
            split_datasets[split] = SPFNDataset(dataset_path, split_path, n_max_instances=256, noisy=True, k_eig=k_eig, input_features=input_features, use_cache=False)
            num_samples = len(split_datasets[split])
            writers[split] = SummaryWriter(save_path / split)
            split_samplers[split] = torch.utils.data.RandomSampler(
                split_datasets[split], replacement=True, num_samples=num_samples
            )

            split_loaders[split] = DataLoader(split_datasets[split], batch_size=batch_size, sampler=split_samplers[split], num_workers=num_workers)

        # === Create the model

        C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

        model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                                C_out=n_class,
                                                C_width=k_eig, 
                                                N_block=n_block,
                                                last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                                outputs_at=output_as,
                                                dropout=True)


        model = model.to(device)

        if args.command != "train":
            # load the pretrained model
            print("Loading pretrained model from: " + str(args.checkpoint_path))
            model.load_state_dict(torch.load(args.checkpoint_path))


        # === Optimize
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


        if args.command == "train":
            train()
        elif args.command == "test":
            print("Testing...")
            evaluate(split_loaders['test'], save_pred=True)
        elif args.command == "evaluate":
            print("Evaluating...")
            test_acc = evaluate(split_loaders['eval'], save_pred=True)
            print("Overall accuracy: {:06.3f}%".format(100*test_acc))

        for k, writer in writers.items():
            if writer is not None:
                writer.close()
