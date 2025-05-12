import os
import cv2
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from torchvision import transforms
from diffdrr.data import read
from einops import rearrange
from torchio import Subject
from diffdrr.drr import DRR
from sys import getsizeof


class Cv2CropBlackAndResize:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img, masks):
        gray = (img.numpy() * 255).astype(np.uint8)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
            
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return TF.resize(img, self.output_size)

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        size = max(w, h)

        cx = x + w // 2
        cy = y + h // 2
        half_size = size // 2

        x1 = max(cx - half_size, 0)
        y1 = max(cy - half_size, 0)
        
        x2 = min(x1 + size, 256)
        y2 = min(y1 + size, 256)
        x1 = max(x2 - size, 0)
        y1 = max(y2 - size, 0)

        cropped_img = img[y1:y2, x1:x2]
        cropped_masks = masks[:, :, y1:y2, x1:x2]

        resized_img = TF.resize(cropped_img.unsqueeze(0), self.output_size).squeeze()
        resized_masks = TF.resize(
            cropped_masks,
            self.output_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        
        resized_np_img = (resized_img.cpu().numpy() * 255).astype(np.uint8)
        equalized = cv2.equalizeHist(resized_np_img)
        resized_img = torch.tensor(equalized)

        return resized_img, resized_masks


def load_ct(
    file_path: str,
    label_path: str,
    labels: int | list = None,
    orientation: str | None = "AP",
    bone_attenuation_multiplier: float = 1.0,
    **kwargs,
) -> Subject:
    """Load an example chest CT for demonstration purposes."""
    volume = file_path
    labelmap = label_path
    return read(
        volume,
        labelmap,
        labels,
        orientation=orientation,
        bone_attenuation_multiplier=bone_attenuation_multiplier,
        **kwargs,
    )


def convert_ct_xray(ct_directory, output_directory):
    label_directory = os.path.join(ct_directory, "segmentations")
    ct_file = os.path.join(ct_directory, "ct.nii.gz")

    # Create a list to store the loaded label images and their names
    label_images = []
    label_names = []
    mem_total = 0.0

    # Load all the labels
    for filename in sorted(os.listdir(label_directory)):
        if filename.endswith(".nii.gz"):
            label_img = nib.load(os.path.join(label_directory, filename))
            label_images.append(label_img.get_fdata())

            mem_total += getsizeof(label_images[-1]) / 1e9
            if mem_total >= 50.0:
                return

            label_names.append(filename.split(".")[0])

    # Assume all labels are the same shape
    shape = label_images[0].shape
    combined_label_data = np.zeros(
        shape, dtype=np.int16
    )  # Use int16 to store label ids

    # Assign a unique integer to each label
    for idx, label_data in enumerate(label_images, start=1):  # Start labeling from 1
        combined_label_data[label_data > 0] = idx

    # Save combined label
    first_label = nib.load(
        os.path.join(label_directory, os.listdir(label_directory)[0])
    )  # Use first label's affine/header
    combined_label_img = nib.Nifti1Image(
        combined_label_data, affine=first_label.affine, header=first_label.header
    )
    combined_label_img.to_filename(os.path.join(ct_directory, "combined_labels.nii.gz"))

    label_index = {idx: name for idx, name in enumerate(label_names, start=1)}
    pd.Series(label_index).to_csv(
        os.path.join(ct_directory, "combined_label_index.csv")
    )

    # Load CT + label data
    subject = load_ct(
        ct_file,
        ct_directory + "/combined_labels.nii.gz",
        labels=None,
        orientation="AP",
        bone_attenuation_multiplier=2.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DRR for CT
    drr = DRR(
        subject,
        sdd=1020.0,
        height=256,
        delx=2.0,
    ).to(device)

    # Define a fixed translation vector (e.g., from side view)
    translations = torch.tensor([[0.0, 800.0, 0.0]], device=device)

    # Define rotations (ZXY Euler) for 4 cardinal views
    cardinal_views = {
        "AP": [0.0, 0.0, 0.0],
        "LAP": [45.0, 0.0, 0.0],
        "RAP": [315.0, 0.0, 0.0],
        "PA": [180.0, 0.0, 0.0],
        "LPA": [225.0, 0.0, 0.0],
        "RPA": [135.0, 0.0, 0.0]
    }

    im_transform = Cv2CropBlackAndResize((256, 256))
    for view_name, degrees in cardinal_views.items():
        rotations = (
            torch.tensor([degrees], device=device) * np.pi / 180.0
        )  # Convert to radians

        img = drr(
            rotations,
            translations,
            parameterization="euler_angles",
            convention="ZXY",
            mask_to_channels=True,
        )

        im = img.sum(dim=1, keepdim=True)
        im = im.squeeze()
        im = (im - im.min()) / (im.max() - im.min())

        label_img = rearrange(img, "c b h w -> b c h w")
        im, label_img = im_transform(im.cpu(), label_img.cpu())

        np.save(
            os.path.join(output_directory, f"{view_name}_labels.npy"),
            label_img.numpy(),
        )
        plt.imsave(
            os.path.join(output_directory, f"{view_name}_base.png"),
            im.numpy(),
            cmap="gray",
            format="png",
        )
