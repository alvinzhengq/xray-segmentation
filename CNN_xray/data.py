from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import random

class XRaySegmentationDataset(Dataset):
    """
    image_dir : contains *.png (or jpg) – each file is a grayscale CXR
    mask_dir  : contains *.npy      – each file is a 118×256×256 binary mask
    """
    def __init__(self, image_dir, mask_dir, transform=False, eval=False):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.eval = eval
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = []
        for img_path in self.image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]        # e.g. "patient123"
            mpath = os.path.join(mask_dir, base + ".npy")
            if os.path.exists(mpath):
                self.mask_paths.append(mpath)
            else:
                raise FileNotFoundError(f"Mask for {base} not found in {mask_dir}")
        assert len(self.image_paths) == len(self.mask_paths)

        # transforms used for **all** samples (resize + to‑tensor + normalize)
        self.to_tensor = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),                                  # (1, H, W) in [0,1]
            transforms.Normalize(mean=[0.485], std=[0.229])         # same stats as training
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ---------- Load image ----------
        img_path = self.image_paths[idx]
        pil_img  = Image.open(img_path).convert("L")   # force single‑channel
        image    = self.to_tensor(pil_img)             # tensor float32, (1,256,256)

        # ---------- Load mask ----------
        mask_path = self.mask_paths[idx]
        mask_np = np.load(mask_path)

        # print(f"Mask shape: {mask_np.shape}")  # Debug print

        if self.eval:
            mask = torch.from_numpy(mask_np)
            mask = mask.permute(2, 0, 1)
            if mask.shape[1:] != (256, 256):
                mask = TF.resize(mask, (256, 256), interpolation=InterpolationMode.NEAREST)

            # print("Eval mask shape: ", mask.shape)
            # print("Eval image shape: ", image.shape)
            return image, mask

        if mask_np.ndim == 4 and mask_np.shape[0] == 1:
            mask_np = mask_np.squeeze(0)
        elif mask_np.ndim == 4 and mask_np.shape[1] == 1:  # Handle case where there's an extra dimension
            mask_np = mask_np.squeeze(1)
        mask_np = (mask_np > 0).astype(np.float32)

        # Ensure mask has 118 channels by padding with zeros if needed
        if mask_np.shape[0] < 118:
            padding = np.zeros((118 - mask_np.shape[0], *mask_np.shape[1:]), dtype=np.float32)
            mask_np = np.concatenate([mask_np, padding], axis=0)
        elif mask_np.shape[0] > 118:
            mask_np = mask_np[:118]

        mask = torch.from_numpy(mask_np)        # (118, 256, 256)

        if mask.shape[1:] != (256, 256):
            mask = TF.resize(mask, (256, 256), interpolation=InterpolationMode.NEAREST)

        if self.transform:

            if random.random() > 0.5:
                image = torch.flip(image, dims=[2])   # flip width
                mask  = torch.flip(mask,  dims=[2])

            # Rotation −10°..+10°
            angle = random.uniform(-10, 10)
            if abs(angle) > 1e-3:
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=InterpolationMode.NEAREST)


            shift = random.uniform(-0.1, 0.1)
            image = torch.clamp(image + shift, 0., 1.)

        mask = (mask > 0.5).float()
        return image, mask
    
class OneSampleDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, mask_path, duplicates=8):
        self.img  = Image.open(img_path).convert("L")
        self.mask = np.load(mask_path).squeeze(0)  # (118, H, W)
        self.dup  = duplicates                     # how many copies
        self.tf   = transforms.Compose([
            transforms.Resize((256,256), InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])
    def __len__(self):
        return self.dup
    def __getitem__(self, idx):
        img = self.tf(self.img)                          # (1,256,256)
        self.mask = (self.mask > 0).astype(np.float32)  # (118,200,200)
        msk = torch.from_numpy(self.mask).float()
        msk = TF.resize(msk, (256,256), InterpolationMode.NEAREST)
        return img, msk
