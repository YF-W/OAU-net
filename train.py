import torch
import matplotlib.pyplot as plt
from model_FCN import FCN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from numpy import *
from model_FCN16s import FCN16s
from model_fcn101 import FCN_res101
from model_unet import UNET
from model_segnet import SegNet
from model_resunet import build_resunet
from model_sharpen_unet import Sharpen_UNET
from model_attention_sharpen import Attention_Sharpen
from module_cbam import CBAMNet
from my_model import my
from model_OAUNet import oaunet
from model_deeplabv3 import DeepLabv3Plus
from model_FCN32s import FCN32s
from model_wyf import YF_2
from model_wyf2 import YF
from model_xy import xy
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    DiceLoss
)
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  # 512 originally
IMAGE_WIDTH = 512  # 512 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/data_lung/train_images"
TRAIN_MASK_DIR = "data/data_lung/train_masks"
VAL_IMG_DIR = "data/data_lung/val_images"
VAL_MASK_DIR = "data/data_lung/val_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    print(f"---Epoch:{epoch}---")
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    epoch_list = []
    acc_list = []
    dice_list = []
    jaccord_list = []
    for epoch in range(NUM_EPOCHS):

        epoch_list.append(epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        # check_accuracy(val_loader, model, device=DEVICE)
        acc, dice, jaccord = check_accuracy(val_loader, model, device=DEVICE)
        acc_list.append(acc)
        dice_list.append(dice)
        jaccord_list.append(jaccord)
        # plt.plot(epoch_list, acc_list)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    plt.plot(epoch_list, jaccord_list)
    # plt.plot(epoch_list, dice_list)
    plt.show()

    print(f"Max Accuracy:{max(acc_list)}")
    print(f"Max Jaccord:{max(jaccord_list)}")
    print(f"Max Dice:{max(dice_list)}")



if __name__ == "__main__":
    main()


