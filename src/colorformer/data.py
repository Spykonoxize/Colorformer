"""Data preprocessing and dataset utilities for Colorformer."""

import glob
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import kornia
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class DataConfig:
    """Configuration for image preprocessing and dataloading."""

    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 16
    max_images: Optional[int] = None


class CinematicPreprocessor:
    """Prepare images for training in Lab color space."""

    def __init__(self, image_size: Tuple[int, int] = (256, 256)) -> None:
        self.pre_transforms = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_tensor = self.pre_transforms(image)
        img_tensor_b = img_tensor.unsqueeze(0)
        lab_tensor = kornia.color.rgb_to_lab(img_tensor_b)

        l_channel = lab_tensor[:, 0:1, :, :]
        l_norm = (l_channel / 50.0) - 1.0

        ab_channel = lab_tensor[:, 1:3, :, :]
        ab_norm = ab_channel / 128.0

        return {
            "L": l_norm.squeeze(0),
            "ab": ab_norm.squeeze(0),
            "original": img_tensor,
        }


class InferencePreprocessor:
    """Prepare images for inference in Lab color space."""

    def __init__(self, image_size: Tuple[int, int] = (256, 256)) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0)
        lab_tensor = kornia.color.rgb_to_lab(img_tensor)

        l_channel = lab_tensor[:, 0:1, :, :]
        l_norm = (l_channel / 50.0) - 1.0

        return {
            "L": l_norm,
            "original": img_tensor,
        }


class SimpleImageFolder(Dataset):
    """Minimal image dataset with recursive file discovery."""

    def __init__(self, folder: str, preprocessor: CinematicPreprocessor, max_images: Optional[int] = None,) -> None:
        all_paths = sorted(
            glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(folder, "**", "*.png"), recursive=True)
            + glob.glob(os.path.join(folder, "**", "*.JPEG"), recursive=True)
            + glob.glob(os.path.join(folder, "**", "*.jpeg"), recursive=True)
        )

        if max_images is not None and max_images < len(all_paths):
            self.paths = all_paths[:max_images]
        else:
            self.paths = all_paths

        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocessor(img)
