import torch
import albumentations as A
import cv2

# Adjust prompt as necessary
class CFG:
    backbone = 'efficientnet-b1'
    num_classes = 1  # Number of output classes
    img_size = [512, 512]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_func = "DiceLoss"

    data_transforms = {
        "train": A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
        ], p=1.0),

        "valid": A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }
