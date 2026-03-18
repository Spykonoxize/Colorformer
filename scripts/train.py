"""Example training script using modular Colorformer package."""

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from colorformer.data import CinematicPreprocessor, SimpleImageFolder
from colorformer.model import ColorFormerUNet, PatchGANDiscriminator
from colorformer.trainer import ColorformerTrainer, TrainerConfig


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = os.environ.get("COLORFORMER_DATASET_DIR", "")
    if not dataset_root:
        raise ValueError(
            "Define COLORFORMER_DATASET_DIR environment variable to the training image folder."
        )

    preproc = CinematicPreprocessor(image_size=(256, 256))
    dataset = SimpleImageFolder(folder=dataset_root, preprocessor=preproc, max_images=3000)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)

    generator = ColorFormerUNet(
        in_channels=1,
        out_channels=2,
        base_channels=32,
        num_heads=(2, 4, 8, 16),
        window_size=7,
    )
    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64, n_layers=3)

    config = TrainerConfig(
        lr_g=1e-4,
        lr_d=2e-4,
        n_critic=3,
        lambda_gan=0.5,
        lambda_l1=100.0,
        lambda_vgg=1000.0,
        lambda_gp=10.0,
        save_dir="outputs/colorformer_training",
        use_amp=True,
    )

    trainer = ColorformerTrainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=str(device),
        config=config,
    )
    trainer.train(num_epochs=30, save_every=5, sample_every=1)


if __name__ == "__main__":
    main()
