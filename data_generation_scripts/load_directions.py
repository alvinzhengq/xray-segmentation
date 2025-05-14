import matplotlib.pyplot as plt
import torch
from diffdrr.drr import DRR
from diffdrr.data import load_example_ct, read
from diffdrr.visualization import plot_drr, plot_mask
import torch.nn.functional as F
import pandas as pd
from torchio import LabelMap, ScalarImage, Subject
import numpy as np
import os
import nibabel as nib

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

def convert_ct_xray(ct_directory, output_directory, translations = torch.tensor([[0.0, 800.0, 0.0]])):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translations = translations.to(device)
    label_directory = os.path.join(ct_directory, "segmentations")
    ct_file = os.path.join(ct_directory, "ct.nii.gz")

    # Create a list to store the loaded label images and their names
    label_images = []
    label_names = []

    # Load all the labels
    for filename in sorted(os.listdir(label_directory)):
        if filename.endswith(".nii.gz"):
            label_img = nib.load(os.path.join(label_directory, filename))
            label_images.append(label_img.get_fdata())
            label_names.append(filename.split(".")[0])

    # Assume all labels are the same shape
    shape = label_images[0].shape
    combined_label_data = np.zeros(shape, dtype=np.int16)  # Use int16 to store label ids

    # Assign a unique integer to each label
    for idx, label_data in enumerate(label_images, start=1):  # Start labeling from 1
        combined_label_data[label_data > 0] = idx

    # Save combined label
    first_label = nib.load(os.path.join(label_directory, os.listdir(label_directory)[0]))  # Use first label's affine/header
    combined_label_img = nib.Nifti1Image(combined_label_data, affine=first_label.affine, header=first_label.header)
    combined_label_img.to_filename(os.path.join(ct_directory, "combined_labels.nii.gz"))

    label_index = {idx: name for idx, name in enumerate(label_names, start=1)}
    pd.Series(label_index).to_csv(os.path.join(ct_directory, "combined_label_index.csv"))


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
        height=200,
        delx=2.0,
    ).to(device)

    # Define rotations (ZXY Euler) for 4 cardinal views
    cardinal_views = {
        "AP": [0.0, 0.0, 0.0], 
        "LLAT": [90.0, 0.0, 0.0],
        "PA": [180.0, 0.0, 0.0],
        "RLAT": [270.0, 0.0, 0.0],
    }

    # Loop through views
    for view_name, degrees in cardinal_views.items():
        rotations = torch.tensor([degrees], device=device) * np.pi / 180.0  # Convert to radians
        
        img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY", mask_to_channels=True)

        im = img.sum(dim=1, keepdim=True)    # DRR image
        #check if first 20 rows of pixels are all 0, if so raise ValueError
        if torch.all(im[0, 0, :20, :] == 0):
            raise ValueError(f"First 20 rows of pixels are all 0 for view {view_name}.")

        label_img = img   # All masks
        # Save label_img as a numPy array
        np.save(os.path.join(output_directory, f"{view_name}_labels.npy"), label_img)
        
        # save im as a png that's 200x200 grayscale
        im = im.squeeze().cpu().numpy()
        im = (im - im.min()) / (im.max() - im.min()) * 255  # Normalize to 0-255
        im = im.astype(np.uint8)
        plt.imsave(os.path.join(output_directory, f"{view_name}_output.png"), im, cmap='gray', format='png')