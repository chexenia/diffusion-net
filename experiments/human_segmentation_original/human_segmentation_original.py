import glob
import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from human_segmentation_original_dataset import HumanSegOrigDataset, ShapifyDataset


# === Options
base_path = os.path.dirname(__file__)

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=["train", "test", "evaluate", "view"])
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
parser.add_argument("--dataset_path", default=os.path.join(base_path, "data/sig17_seg_benchmark"))
args = parser.parse_args()


# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 8

# model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# training settings
train = args.command == "train"
n_epoch = 200
lr = 1e-3
decay_every = 50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')



# Important paths
op_cache_dir = os.path.join(base_path, "data", "op_cache")
pretrain_path = os.path.join(base_path, "pretrained_models/human_seg_{}_4x128.pth".format(input_features))
model_save_path = os.path.join(base_path, "data/saved_models/human_seg_{}_4x128.pth".format(input_features))
dataset_path = args.dataset_path


# === Load datasets

# Load the evaluate dataset
if args.command == "evaluate":
    eval_dataset = HumanSegOrigDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    eval_loader = DataLoader(eval_dataset, batch_size=None)

if args.command == "test":
    test_dataset = ShapifyDataset(dataset_path, train=False, k_eig=k_eig)
    test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
if args.command == "train":
    train_dataset = HumanSegOrigDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128, 
                                          N_block=4, 
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='faces', 
                                          dropout=True)


model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))


# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(epoch):

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
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        
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
    return train_acc


# Do an evaluation pass on the test dataset 
def evaluate():
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for data in tqdm(eval_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

            # Move to device
            verts = verts.to(device)
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

            # track accuracy
            pred_labels = torch.max(preds, dim=1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            this_num = labels.shape[0]
            correct += this_correct
            total_num += this_num

    test_acc = correct / total_num
    return test_acc 

def test():
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, mpath = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # track accuracy
            pred_labels = torch.max(preds, dim=1).indices.cpu().numpy()

            np.savetxt(Path(mpath).with_suffix(f'.diffnet.{input_features}.txt'), pred_labels, fmt='%d')


if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_acc = train_epoch(epoch)
        test_acc = test()
        print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))

    print(" ==> saving last model to " + model_save_path)
    torch.save(model.state_dict(), model_save_path)

if args.command == "evaluate":
    print("Evaluating...")
    test_acc = evaluate()
    print("Overall accuracy: {:06.3f}%".format(100*test_acc))

if args.command == "test":
    print("Testing...")
    test()

if args.command == "view":
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