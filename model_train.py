import os
import torch
from monai.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from monai.losses.dice import DiceCELoss
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from vit_pytorch.vit_3d import ViT
# from networks.nets import SwinUNETR, UNETR, UNet
from monai.utils import set_determinism
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsChannelLastd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
    CropForegroundd
)
from utils import (
    ConvertToMultiChannelBasedOnBratsClassesd,
    sec_to_minute,
    LinearWarmupCosineAnnealingLR,
)
from lovasz_losses import *
import glob
import argparse
import time
import tqdm
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    gamma = 0.7
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Transformer segmentation pipeline")
    parser.add_argument("--datapath", default="../Dataset_BRATS_2020/Training/", type=str, help="Dataset path")
    parser.add_argument("--epochs", default=5, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--dataset", default="2020", type=str, help="Dataset to use")
    parser.add_argument("--augmented", default=0, type=int, help="Add augmented Dataset")
    parser.add_argument("--seed", default=10, type=int, help="Seed for controlling randomness")
    parser.add_argument("--model", default="swinunetr", type=str, help="Model Name")
    parser.add_argument("--val_frac", default=0.05, type=float, help="fraction of data to use as validation")
    parser.add_argument("--num_heads", default=12, type=int, help="Number of heads to use")
    parser.add_argument("--embed_dim", default=768, type=int, help="Embedding dimension")
    parser.add_argument("--num_worker", default=2, type=int, help="Number of workers for Dataloader")
    parser.add_argument("--weighted_class", default=0, type=int, help="Use weights for classes")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning Rate")
    parser.add_argument("--resume_ckpt", default=1, type=int, help="resume training from pretrained checkpoint")

    args = parser.parse_args()

    seed = args.seed
    set_determinism(seed=seed)

    device = torch.device("cuda")

    ds = args.dataset
    aug = True if args.augmented == 1 else 0
    frac = args.val_frac
    max_epochs = args.epochs
    batch_size = args.batch_size
    model_name = args.model
    num_heads = args.num_heads
    embed_dim = args.embed_dim
    n_workers = args.num_worker
    weighted_class = True if args.augmented == 1 else 0
    lr = args.lr
    resume_ckpt = True if args.resume_ckpt == 1 else 0

    roi_size = [128, 128, 64]
    pixdim = (1.5, 1.5, 2.0)

    best_metric = -1
    best_metric_epoch = -1

    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    if ds == "2020":
        data_dir = args.datapath
        t1_list = sorted(glob.glob(data_dir + "*/*t1.nii"))
        t2_list = sorted(glob.glob(data_dir + "*/*t2.nii"))
        t1ce_list = sorted(glob.glob(data_dir + "*/*t1ce.nii"))
        flair_list = sorted(glob.glob(data_dir + "*/*flair.nii"))
        seg_list = sorted(glob.glob(data_dir + "*/*seg.nii"))
        
        if aug:
            data_dir = "../Dataset_BRATS_2020/Augmented3/"
            t1_list += sorted(glob.glob(data_dir + "*/*t1.nii"))
            t2_list += sorted(glob.glob(data_dir + "*/*t2.nii"))
            t1ce_list += sorted(glob.glob(data_dir + "*/*t1ce.nii"))
            flair_list += sorted(glob.glob(data_dir + "*/*flair.nii"))
            seg_list += sorted(glob.glob(data_dir + "*/*seg.nii"))

    elif ds == "2021":
        data_dir = "../Dataset_BRATS_2021/"
        t1_list = sorted(glob.glob(data_dir + "*/*t1.nii"))
        t2_list = sorted(glob.glob(data_dir + "*/*t2.nii"))
        t1ce_list = sorted(glob.glob(data_dir + "*/*t1ce.nii"))
        flair_list = sorted(glob.glob(data_dir + "*/*flair.nii"))
        seg_list = sorted(glob.glob(data_dir + "*/*seg.nii"))

    elif ds == "2020-2021":  # combiantion of 2020 and 2021, TODO: remove
        data_dir = "../Dataset_BRATS_2020/Training/"
        t1_list = sorted(glob.glob(data_dir + "*/*t1.nii"))
        t2_list = sorted(glob.glob(data_dir + "*/*t2.nii"))
        t1ce_list = sorted(glob.glob(data_dir + "*/*t1ce.nii"))
        flair_list = sorted(glob.glob(data_dir + "*/*flair.nii"))
        seg_list = sorted(glob.glob(data_dir + "*/*seg.nii"))
        data_dir = "../Dataset_BRATS_2021/"
        t1_list += sorted(glob.glob(data_dir + "*/*t1.nii"))
        t2_list += sorted(glob.glob(data_dir + "*/*t2.nii"))
        t1ce_list += sorted(glob.glob(data_dir + "*/*t1ce.nii"))
        flair_list += sorted(glob.glob(data_dir + "*/*flair.nii"))
        seg_list += sorted(glob.glob(data_dir + "*/*seg.nii"))


    n_data = len(t1_list)
    print(f"Dataset size: {n_data}")
    data_dicts = [
        {"images": [t1, t2, t1ce, f], "label": label_name}
        for t1, t2, t1ce, f, label_name in zip(
            t1_list, t2_list, t1ce_list, flair_list, seg_list
        ) 
        # if "195" not in t1 if "168" not in t1
    ]

    val_files, train_files = (
        data_dicts[: int(n_data * frac)],
        data_dicts[int(n_data * frac):],
    )

    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["images", "label"]),
            AsChannelFirstd(keys="images", channel_dim=0),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(
                keys=["images", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["images", "label"], axcodes="RAS"),
            RandSpatialCropd(
                keys=["images", "label"], roi_size=roi_size, random_size=False
            ),
            RandFlipd(keys=["images", "label"], prob=0.5, spatial_axis=0),
            NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="images", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="images", offsets=0.1, prob=0.5),
            ToTensord(keys=["images", "label"]),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["images", "label"]),
            AsChannelFirstd(keys="images", channel_dim=0),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(
                keys=["images", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["images", "label"], axcodes="RAS"),
            CenterSpatialCropd(keys=["images", "label"], roi_size=roi_size),
            NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
            ToTensord(keys=["images", "label"]),
        ]
    )
    

    train_ds = Dataset(data=train_files, transform=train_transform)
    val_ds = Dataset(data=val_files, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    print(train_loader)
    if weighted_class:
        weights = np.array([45, 16, 50], dtype="f")
        class_weights = torch.tensor(
            weights, dtype=torch.float32, device=torch.device("cuda:0")
        )
    else:
        class_weights = None

    in_ch = 4
    out_ch = 3

    model = ViT(
        image_size = 128,          # image size
        frames = 64,               # number of frames
        image_patch_size = 16,     # image patch size
        frame_patch_size = 2,      # frame patch size
        num_classes = 4,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        channels = 4,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)
    print(model.parameters())
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    loss_function = FocalLoss(gamma=1)
    # lossfunc = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # # # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, nesterov=True, weight_decay=1e-5)

    # scheduler = LinearWarmupCosineAnnealingLR(
    #     optimizer, warmup_epochs=1, max_epochs=max_epochs
    # )
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    torch.cuda.empty_cache()

    results_path = os.path.join(".", "RESULTS")
    if os.path.exists(results_path) == False:
        os.mkdir(results_path)
    for epoch in range(max_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        train_tqdm = tqdm.tqdm(train_loader)
        for batch_data in train_tqdm:
            inputs, label = (
            batch_data["images"].to(device),
            batch_data["label"].to(device),
            )
            print(inputs.shape)
            output = model(inputs)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )