"""Training utilities for Colorformer GAN."""

import os
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import ColorformerLoss


@dataclass
class TrainerConfig:
    """Hyperparameters for Colorformer training."""

    lr_g: float = 1e-4
    lr_d: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    n_critic: int = 5
    lambda_gan: float = 0.5
    lambda_l1: float = 100.0
    lambda_vgg: float = 1000.0
    lambda_gp: float = 10.0
    use_amp: bool = True
    save_dir: str = "outputs/training"


class ColorformerTrainer:
    """Train Colorformer generator and PatchGAN discriminator."""

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        dataloader,
        device: str = "cuda",
        config: TrainerConfig | None = None,
    ) -> None:
        if config is None:
            config = TrainerConfig()

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader = dataloader
        self.device = device
        self.n_critic = config.n_critic
        self.save_dir = config.save_dir
        self.start_epoch = 0

        device_str = str(device) if not isinstance(device, str) else device
        self.use_amp = config.use_amp and ("cuda" in device_str)

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "samples"), exist_ok=True)

        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.lr_g,
            betas=config.betas,
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_d,
            betas=config.betas,
        )

        self.loss_fn = ColorformerLoss(
            lambda_gan=config.lambda_gan,
            lambda_l1=config.lambda_l1,
            lambda_vgg=config.lambda_vgg,
            lambda_gp=config.lambda_gp,
            device=device,
        )

        self.scaler_g = GradScaler("cuda", enabled=self.use_amp)
        self.scaler_d = GradScaler("cuda", enabled=self.use_amp)

        self.history: Dict[str, list[float]] = {
            "g_loss": [],
            "d_loss": [],
        }

    def train_discriminator_step(
        self,
        l_channel: torch.Tensor,
        real_ab: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.optimizer_d.zero_grad()

        with torch.no_grad():
            with autocast("cuda", enabled=self.use_amp):
                fake_ab = self.generator(l_channel)

        with autocast("cuda", enabled=self.use_amp):
            d_real = self.discriminator(l_channel, real_ab)
            d_fake = self.discriminator(l_channel, fake_ab.detach())
            loss_wgan = d_fake.mean() - d_real.mean()

        gradient_penalty = self.loss_fn.gradient_penalty(
            self.discriminator,
            real_ab,
            fake_ab.detach(),
            l_channel,
        )
        loss_total = loss_wgan + self.loss_fn.lambda_gp * gradient_penalty

        self.scaler_d.scale(loss_total).backward()
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()

        return {
            "total": loss_total,
            "wgan": loss_wgan,
            "gp": gradient_penalty,
            "d_real": d_real.mean(),
            "d_fake": d_fake.mean(),
        }

    def train_generator_step(
        self,
        l_channel: torch.Tensor,
        real_ab: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        self.optimizer_g.zero_grad()

        with autocast("cuda", enabled=self.use_amp):
            fake_ab = self.generator(l_channel)
            d_fake = self.discriminator(l_channel, fake_ab)

            loss_gan = -d_fake.mean()
            loss_l1 = self.loss_fn.l1_loss(fake_ab, real_ab)

            fake_rgb = self.loss_fn.lab_to_rgb(l_channel, fake_ab)
            real_rgb = self.loss_fn.lab_to_rgb(l_channel, real_ab)
            loss_vgg = self.loss_fn.vgg_loss(fake_rgb, real_rgb)

            loss_total = (
                self.loss_fn.lambda_gan * loss_gan
                + self.loss_fn.lambda_l1 * loss_l1
                + self.loss_fn.lambda_vgg * loss_vgg
            )

        self.scaler_g.scale(loss_total).backward()
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        return {
            "total": loss_total,
            "gan": loss_gan,
            "l1": loss_l1,
            "vgg": loss_vgg,
        }, fake_ab

    def train_epoch(self, epoch: int, num_epochs: int) -> tuple[float, float]:
        self.generator.train()
        self.discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            l_channel = batch["L"].to(self.device)
            real_ab = batch["ab"].to(self.device)

            d_loss_dict = None
            for _ in range(self.n_critic):
                d_loss_dict = self.train_discriminator_step(l_channel, real_ab)

            g_loss_dict, _ = self.train_generator_step(l_channel, real_ab)

            epoch_g_loss += g_loss_dict["total"].item()
            epoch_d_loss += d_loss_dict["total"].item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "G_loss": f"{g_loss_dict['total'].item():.4f}",
                    "D_loss": f"{d_loss_dict['total'].item():.4f}",
                }
            )

        avg_g_loss = epoch_g_loss / max(num_batches, 1)
        avg_d_loss = epoch_d_loss / max(num_batches, 1)

        self.history["g_loss"].append(avg_g_loss)
        self.history["d_loss"].append(avg_d_loss)

        return avg_g_loss, avg_d_loss

    def save_samples(self, epoch: int, num_samples: int = 4) -> None:
        self.generator.eval()

        batch = next(iter(self.dataloader))
        available_samples = batch["L"].shape[0]
        sample_count = min(num_samples, available_samples)
        if sample_count == 0:
            self.generator.train()
            return

        l_channel = batch["L"][:sample_count].to(self.device)
        original = batch["original"][:sample_count]

        with torch.no_grad():
            with autocast("cuda", enabled=self.use_amp):
                fake_ab = self.generator(l_channel)

        fake_rgb = self.loss_fn.lab_to_rgb(l_channel, fake_ab).cpu()

        fig, axes = plt.subplots(sample_count, 3, figsize=(12, 4 * sample_count))
        if sample_count == 1:
            axes = axes.reshape(1, -1)

        for i in range(sample_count):
            l_display = (l_channel[i, 0].cpu() + 1) / 2
            axes[i, 0].imshow(l_display, cmap="gray")
            axes[i, 0].set_title("Input (Grayscale)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(fake_rgb[i].permute(1, 2, 0).numpy())
            axes[i, 1].set_title("Colorized")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(original[i].permute(1, 2, 0).numpy())
            axes[i, 2].set_title("GT")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, "samples", f"epoch_{epoch + 1:03d}.png"),
            dpi=150,
        )
        plt.close()

        self.generator.train()

    def save_checkpoint(self, epoch: int) -> str:
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "scaler_g_state_dict": self.scaler_g.state_dict(),
            "scaler_d_state_dict": self.scaler_d.state_dict(),
            "history": self.history,
            "use_amp": self.use_amp,
        }
        path = os.path.join(
            self.save_dir,
            "checkpoints",
            f"checkpoint_epoch_{epoch + 1:03d}.pt",
        )
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

        if "scaler_g_state_dict" in checkpoint:
            self.scaler_g.load_state_dict(checkpoint["scaler_g_state_dict"])
            self.scaler_d.load_state_dict(checkpoint["scaler_d_state_dict"])

        self.history = checkpoint["history"]
        self.start_epoch = checkpoint["epoch"] + 1
        return checkpoint["epoch"]

    def train(self, num_epochs: int, save_every: int = 5, sample_every: int = 1) -> None:
        for epoch in range(self.start_epoch, num_epochs):
            avg_g_loss, avg_d_loss = self.train_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}/{num_epochs} | G_loss={avg_g_loss:.4f} D_loss={avg_d_loss:.4f}")

            if (epoch + 1) % sample_every == 0:
                self.save_samples(epoch)

            if (epoch + 1) % save_every == 0:
                checkpoint_path = self.save_checkpoint(epoch)
                print(f"Checkpoint saved: {checkpoint_path}")

        final_path = self.save_checkpoint(num_epochs - 1)
        print(f"Training complete. Last checkpoint: {final_path}")
