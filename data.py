import numpy as np
import torch
import cv2

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

    structure: str
    view: str


class SegmentationDataset(Dataset):
    def __init__(self, data: List[Datapoint]):
        self.data = data
        self.count = len(data)
        self.lang = Language()

    def idx_from_sentence(self, sentence):
        self.lang.add_sentence(sentence)
        return [self.lang.word2index[word] for word in sentence.split(" ")]

    def tensor_from_sentence(self, sentence):
        indexes = self.idx_from_sentence(sentence)
        return torch.tensor(indexes, dtype=torch.long, device="cpu")

    def sentence_from_tensor(self, tensor):
        indexes = tensor.tolist()
        return " ".join([self.lang.index2word[idx] for idx in indexes])

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        point = self.data[idx]

        prompt = f"segmentation mask of {point.structure} of {'an' if point.view[0] in 'aeiou' else 'a'} {point.view} view xray"
        prompt = self.tensor_from_sentence(prompt)

        return (
            prompt,
            torch.from_numpy(point.base_img),
            torch.from_numpy(point.label_binary),
            torch.from_numpy(point.label_count),
            torch.from_numpy(point.gt_mask)
        )


def image_statistics(images: np.ndarray, debug=True):
    """Run this function once on your dataset before implementing preprocessing"""
    all_min, all_max = images.min(), images.max()
    overall_mean = images.mean()

    all_values = images.flatten()
    p1, p99 = np.percentile(all_values, [1, 99])

    if debug:
        print(
            f"1st percentile: {p1}, 99th percentile: {p99}, Global min: {all_min}, Global max: {all_max}, Mean: {overall_mean}"
        )

    return p1, p99


def generate_labels(mask, patch_size=8):
    _, h, w = mask.shape
    patch_h, patch_w = h // patch_size, w // patch_size

    patch_binary = np.zeros((patch_h, patch_w))
    patch_count = np.zeros((patch_h, patch_w))

    for i in range(patch_h):
        for j in range(patch_w):
            patch = mask[
                ...,
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            
            patch_binary[i, j] = 1 if np.count_nonzero(patch) > (0.80 * patch_size**2) else 0
            patch_count[i, j] = np.count_nonzero(patch.flatten())

    return patch_binary, patch_count


def normalize_images(images: np.ndarray):
    p1, p99 = image_statistics(images[np.nonzero(images)], debug=False)

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
