import numpy as np
import random
import torch
import cv2

from scipy import ndimage
from typing import List
from dataclasses import dataclass
from torch.utils.data import Dataset


class Language:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.count = 0

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.count
            self.word2count[word] = 1
            self.index2word[self.count] = word
            self.count += 1
        else:
            self.word2count[word] += 1


@dataclass
class Datapoint:
    base_img: np.ndarray
    gt_mask: np.ndarray
    label_binary: np.ndarray
    label_count: np.ndarray
    mask_center: np.ndarray

    structure: str
    view: str


class SegmentationDataset(Dataset):
    def __init__(self, data: List[Datapoint]):
        self.data = data
        self.count = len(data)
        self.lang = Language()

        self.left_lung = [
            i for i, p in enumerate(self.data) if p.structure == "left lung"
        ]
        self.right_lung = [
            i for i, p in enumerate(self.data) if p.structure == "right lung"
        ]
        self.heart = [i for i, p in enumerate(self.data) if p.structure == "heart"]

    def idx_from_sentence(self, sentence):
        self.lang.add_sentence(sentence)
        return [self.lang.word2index[word] for word in sentence.split(" ")] + [
            -1 for _ in range(max(0, 32 - len(sentence.split(" "))))
        ]

    def tensor_from_sentence(self, sentence):
        indexes = self.idx_from_sentence(sentence)
        return torch.tensor(indexes, dtype=torch.long, device="cpu")

    def sentence_from_tensor(self, tensor):
        indexes: List = tensor.tolist()
        indexes = indexes[: indexes.index(-1)]
        return " ".join([self.lang.index2word[idx] for idx in indexes])

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        point = self.data[idx]

        prompt = f"{point.structure} in {'an' if point.view[0] in 'aeiou' else 'a'} {point.view} view xray"
        prompt = self.tensor_from_sentence(prompt)

        return (
            prompt,
            torch.from_numpy(point.base_img),
            torch.from_numpy(point.label_binary),
            torch.from_numpy(point.label_count),
            torch.from_numpy(point.gt_mask),
            torch.from_numpy(point.mask_center),
        )

    def get_random(self, structure=None):
        if structure is None:
            return self.__getitem__(random.randrange(0, self.count))

        elif structure == "left lung":
            return self.__getitem__(random.choice(self.left_lung))

        elif structure == "right lung":
            return self.__getitem__(random.choice(self.right_lung))

        elif structure == "lung":
            return self.__getitem__(random.choice(self.left_lung + self.right_lung))

        elif structure == "heart":
            return self.__getitem__(random.choice(self.heart))

        return self.__getitem__(random.randrange(0, self.count))

    def get_random_f(self, f=lambda _: True):
        idxs = [i for i, p in enumerate(self.data) if f(p)]
        return self.__getitem__(random.choice(idxs))


###################################################################################################


def generate_labels(mask, patch_size=8):
    if len(mask.shape) == 3:
        mask = np.expand_dims(mask, axis=0)
        B = 1
    else:
        B = mask.shape[0]

    h, w = mask.shape[-2:]
    patch_h, patch_w = h // patch_size, w // patch_size

    patch_binary = np.zeros((B, patch_h, patch_w))
    patch_count = np.zeros((B, patch_h, patch_w))

    for b in range(B):
        for i in range(patch_h):
            for j in range(patch_w):
                patch = mask[
                    b,
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]

                patch_binary[b, i, j] = (
                    1 if np.count_nonzero(patch) > (0.75 * patch_size**2) else 0
                )
                patch_count[b, i, j] = np.count_nonzero(patch.flatten())

    if B == 1:
        patch_binary = patch_binary.squeeze(0)
        patch_count = patch_count.squeeze(0)

    return patch_binary, patch_count


def image_statistics(images: np.ndarray, debug=True):
    all_min, all_max = images.min(), images.max()
    overall_mean = images.mean()

    all_values = images.flatten()
    p1, p99 = np.percentile(all_values, [1, 99])

    if debug:
        print(
            f"1st percentile: {p1}, 99th percentile: {p99}, Global min: {all_min}, Global max: {all_max}, Mean: {overall_mean}"
        )

    return p1, p99


def find_mask_center(binary_mask):
    moments = cv2.moments(binary_mask.astype(np.uint8))

    if moments["m00"] != 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        return np.array([center_x, center_y], dtype=np.float32)
    else:
        return None


def normalize_images(images: np.ndarray):
    if np.count_nonzero(images) == 0:
        return None

    p1, p99 = image_statistics(images[np.nonzero(images)], debug=False)
    if p99 == 0 or p1 == p99:
        return None

    new_images = np.clip(images, p1, p99)
    new_images = (new_images - p1) / (p99 - p1)

    return new_images


def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_new = clahe.apply(img.astype(np.uint8))

    return img_new


def enhance_edges(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
    sharpened = cv2.addWeighted(img, 1.3, blurred, -0.3, 0)

    return sharpened[..., np.newaxis]


class RemoveWhiteLines(torch.nn.Module):
    def __init__(self, dark=0.1, bright=0.8):
        super().__init__()
        self.dark = dark
        self.bright = bright

    def forward(self, img):
        img_np = img.numpy().squeeze(0)

        dark_regions = img_np < self.dark
        bright_lines = img_np > self.bright

        target_pixels = bright_lines & ndimage.binary_dilation(dark_regions)
        img[0][target_pixels] = 0.0

        return img
