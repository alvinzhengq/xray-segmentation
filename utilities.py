from typing import Tuple

import torch


class ImageGaussianDistribution:
    def __init__(
        self, image_shape: Tuple[int, int, int], mean: float = 0.0, std: float = 1.0
    ) -> None:
        self.image_shape = image_shape
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int, device: str = 'cpu'):
        C, H, W = self.image_shape
        return torch.randn(batch_size, C, H, W, device=device) * self.std + self.mean
