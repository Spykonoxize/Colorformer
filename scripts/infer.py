"""Example inference script using modular Colorformer package."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from colorformer.data import InferencePreprocessor
from colorformer.inference import colorize_image
from colorformer.model import ColorFormerUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Colorformer inference on one image")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="outputs/inference_result.png", help="Output visualization path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ColorFormerUNet(
        in_channels=1,
        out_channels=2,
        base_channels=32,
        num_heads=(2, 4, 8, 16),
        window_size=7,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()

    preprocessor = InferencePreprocessor(image_size=(256, 256))
    result = colorize_image(args.image, model, preprocessor, device)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(result["grayscale"], cmap="gray")
    axes[0].set_title("Entree (N&B)")
    axes[0].axis("off")

    axes[1].imshow(result["colorized"].permute(1, 2, 0).numpy())
    axes[1].set_title("Colorisation")
    axes[1].axis("off")

    axes[2].imshow(result["original"].permute(1, 2, 0).numpy())
    axes[2].set_title("Original")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Inference saved to: {args.output}")


if __name__ == "__main__":
    main()
