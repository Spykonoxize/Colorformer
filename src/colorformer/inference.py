"""Inference helpers for Colorformer models."""

from dataclasses import dataclass
from typing import Dict, Tuple

import kornia
import torch
from PIL import Image
from torch.amp import autocast
from .data import InferencePreprocessor


@dataclass
class InferenceConfig:
    """Configuration used to preprocess inputs for inference."""

    image_size: Tuple[int, int] = (256, 256)
    use_amp: bool = True


def colorize_image(image_path: str, model: torch.nn.Module, preprocessor: InferencePreprocessor, device: torch.device, use_amp: bool = True,) -> Dict[str, torch.Tensor]:
    """Run model inference on one image and return display-ready tensors.

    The model is expected to predict normalized ab channels in [-1, 1].
    """

    image = Image.open(image_path).convert("RGB")
    data = preprocessor(image)
    l_tensor = data["L"].to(device)

    # AMP is enabled only on CUDA to keep CPU execution stable.
    amp_enabled = use_amp and ("cuda" in str(device))
    with torch.no_grad():
        with autocast("cuda", enabled=amp_enabled):
            pred_ab = model(l_tensor)

    l_denorm = (l_tensor + 1.0) * 50.0
    ab_denorm = pred_ab * 128.0
    lab_pred = torch.cat([l_denorm, ab_denorm], dim=1)
    rgb_pred = kornia.color.lab_to_rgb(lab_pred).clamp(0, 1)

    return {
        "grayscale": (l_tensor[0, 0].cpu() + 1) / 2,
        "colorized": rgb_pred[0].cpu(),
        "original": data["original"][0],
    }
