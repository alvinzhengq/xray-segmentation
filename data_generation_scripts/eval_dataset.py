

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import re

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def draw_organ(ax, array, color='b'):
    for i in range(array.shape[0]):
        x, y = array[i, :]
        circ = plt.Circle((x, y), radius=3, color=color, fill=True)
        ax.add_patch(circ)

def draw_lines(ax, array, color='b'):
    for i in range(1, array.shape[0]):
        x1, y1 = array[i - 1, :]
        x2, y2 = array[i, :]
        ax.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1)

def draw_organs(ax, RL, LL, H=None, RCLA=None, LCLA=None, img=None):
    if img is not None:
        ax.imshow(img, cmap='gray')
    else:
        img = np.zeros([1024, 1024])
        ax.imshow(img)
    ax.axis('off')

    draw_lines(ax, RL, 'r')
    draw_lines(ax, LL, 'g')
    draw_organ(ax, RL, 'r')
    draw_organ(ax, LL, 'g')

    if H is not None:
        draw_lines(ax, H, 'y')
        draw_organ(ax, H, 'y')

    if RCLA is not None:
        draw_lines(ax, RCLA, 'purple')
        draw_organ(ax, RCLA, 'purple')

    if LCLA is not None:
        draw_lines(ax, LCLA, 'purple')
        draw_organ(ax, LCLA, 'purple')

def landmarks_to_mask(shape, *landmark_arrays):
    mask = np.zeros(shape, dtype=np.uint8)
    for i, landmark in enumerate(landmark_arrays):
        mask_2d = np.zeros(shape[:2], dtype=np.uint8)
        if landmark is not None and len(landmark) > 2:
            pts = landmark.astype(np.int32).reshape((-1, 1, 2))
            if not np.allclose(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[0][None, :]])  # Close polygon
            cv2.fillPoly(mask_2d, [pts], color=1)
        # Add the filled polygon to the 3rd dimension of the mask
        mask[:, :, i] = mask_2d
    return mask

# Load image paths
images_directory = pathlib.Path("Chest-xray-landmark-dataset/Images/")
all_files = sorted([str(f) for f in images_directory.glob("*.png")], key=natural_key)
output_directory = "eval_dataset_output"
for image_file in all_files:
    RL_path = image_file.replace("Images", "landmarks/RL").replace(".png", ".npy")
    LL_path = image_file.replace("Images", "landmarks/LL").replace(".png", ".npy")
    H_path = image_file.replace("Images", "landmarks/H").replace(".png", ".npy")

    # Load image and landmarks
    img = cv2.imread(image_file, 0)
    try:
        RL = np.load(RL_path)
        LL = np.load(LL_path)
        H = np.load(H_path)

        # Generate filled mask
        # Generate filled mask
        mask = landmarks_to_mask((1024, 1024, 3), RL, LL, H)
        #resize mask to 200x200x3
        downsized_mask = np.stack([
                        cv2.resize(mask[:, :, i], (200, 200), interpolation=cv2.INTER_NEAREST)
                        for i in range(mask.shape[2])
                        ], axis=2)
        img_resized = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
        # Save the resized image as a png and the mask as a numpy array
        os.makedirs(output_directory, exist_ok=True)
        cv2.imwrite(os.path.join(output_directory, os.path.basename(image_file)), img_resized)
        np.save(os.path.join(output_directory, os.path.basename(image_file).replace(".png", "_mask.npy")), downsized_mask)
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        continue

