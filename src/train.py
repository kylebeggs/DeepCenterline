#%%
from torch.utils.tensorboard import SummaryWriter
import os, glob
import logging
import time
import argparse
from natsort import natsorted

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import torch
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)

from utils import compute_euclidean_distance_map
from models import UNetPooling, UNetStriding

root_dir = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help='Name the model you are training.')
    parser.add_argument('--epochs', default=100, type=int , help='Set max number of epochs.')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='Set the initial learning rate to be used with the reduced step scheduler.')
    parser.add_argument('--model', type=str, default='pooling', help='Set which model to use. Options are pooling or striding.')

    args = parser.parse_args()

    return args

if torch.cuda.is_available():
	os.environ["CUDA_VISIBILE_DEVICES"] = "0,1"
	device = torch.device("cuda")
	num_gpus = torch.cuda.device_count()
	print(f"\nUsing {num_gpus} gpus.\n\n")
	workers = 2*num_gpus
else:
	device = torch.device("cpu")
	num_gpus = 0
	print("\nUsing ", device)
	workers = 1

#seg = nib.load("../data/DeepVesselNet/seg/1.nii.gz")
#cen = nib.load("../data/DeepVesselNet/centerline/1.nii.gz")
#map = compute_euclidean_distance_map(seg,cen)
#
#plt.figure(figsize=(12,12))
#plt.imshow(map[:,:,50], vmax = np.max(map))
#plt.colorbar()
#plt.show()



#%%

# when running as main
args = parse_args()	
run_name = args.name
model_type = args.model
learning_rate = args.lr
max_epochs = args.epochs

# jupyter
#run_name = "unet-striding-residualV2"
#model_type = "striding"
#learning_rate = 1e-3
#max_epochs = 100

# dataloader
batch_size = 4
   
# training
decay_learning_rate = True
   
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True
USE_AMP = False

#%%
	
logging.basicConfig(filename="../logs/"+run_name+".log",
                    filemode='a',
                    format='%(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
   
# print info to screen
logging.info("Running "+run_name)
logging.info(f"  learning rate: {learning_rate}")
logging.info(f"  batch size: {batch_size}")


#%% ---------------------------------------------------------------------------
# load images from data directories

#!! Load training images and targets into a dictionary
segs = natsorted(glob.glob(os.path.join(root_dir,"../data/DeepVesselNet/seg/*.nii.gz")))
maps = natsorted(glob.glob(os.path.join(root_dir,"../data/DeepVesselNet/map/*.nii.gz")))

data = [{"image": seg, "label": map} for seg, map in zip(segs,maps)]

class SyntheticVesselDataset(Dataset):
    def __init__(self, img_label_list, transform=None):
        self.img_label_list = img_label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, idx):
        if self.transform:
            img_label = self.transform(self.img_label_list[idx])
        else:
            img_label = self.img_label_list[idx]
        return img_label


#%%
# transforms

train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image","label"], roi_size=(128,128,128), random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image","label"], roi_size=(128,128,128), random_size=False),
        EnsureTyped(keys=["image", "label"]),
    ]
)

inference_transform = Compose(
    [EnsureType()]
)

#%%
# instantiate model, loss, and inference

if model_type=="striding":
    model = UNetStriding(feat_channels=[8,32,32,64,128])
elif model_type=="pooling":
    model = UNetPooling(feat_channels=[8,32,32,64,128])
else:
    raise ValueError(f"Did not choose a valid type of model. Options are striding or pooling.")
    

if num_gpus > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.to(device)
logging.info(model)

def masked_mse_loss(predict, target, mask):
    loss = torch.sum(((predict-target)*mask)**2.0)  / torch.sum(mask)
    return loss

loss_function = torch.nn.MSELoss()



#%%
# create dataloaders

train_size = int(0.8*len(data))
val_size = len(data) - train_size
train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

train_dataset = SyntheticVesselDataset(train_data, transform=train_transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available()
)

val_dataset = SyntheticVesselDataset(val_data, transform=val_transform)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available()
)


#%%
# train

writer = SummaryWriter('../runs/' + run_name)

# use amp to accelerate training
if USE_AMP:
    scaler = torch.cuda.amp.GradScaler()

optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

start = time.time()
for epoch in range(max_epochs):
    model.train()
    current_learning_rate = optimizer.param_groups[0]["lr"]

    # perform epoch training step
    train_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        train_images, train_maps = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()

        if USE_AMP:
            with torch.cuda.amp.autocast():
                train_outputs = model(train_images)
                #loss = loss_function(train_outputs, train_maps, train_images)
                loss = loss_function(train_outputs, train_maps)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            train_outputs = model(train_images)
            #loss = loss_function(train_outputs, train_maps, train_images)
            loss = loss_function(train_outputs, train_maps)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

    # calculate metrics
    train_loss /= step

    writer.add_scalar("training loss", train_loss, epoch + 1)
    writer.add_scalar("learning rate", current_learning_rate, epoch + 1)

    if decay_learning_rate:
        lr_scheduler.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        step = 0
        for val_data in val_loader:
            step += 1
            val_images, val_maps = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = model(val_images)
            #loss = loss_function(val_outputs, val_maps, val_images)
            loss = loss_function(val_outputs, val_maps)
            val_loss += loss.item()
        
    val_loss /= step
    writer.add_scalar("validation loss", train_loss, epoch + 1)
    
    logging.info(f"[{epoch+1:3d}/{max_epochs:3d}] train loss: {train_loss:.3e} val loss: {val_loss:.3e}")
    print(f"[{epoch+1:3d}/{max_epochs:3d}] train loss: {train_loss:.3e} val loss: {val_loss:.3e}\n")

    if epoch == 0:
        prev_best = val_loss
    if val_loss < prev_best:
        save_path = "../trained_models/" + run_name + ".pth"
        torch.save(model.state_dict(), save_path)
        prev_best = val_loss


end = time.time()
logging.info(f"Training took {end-start:.1f} seconds.")
writer.flush()
writer.close()


#%%