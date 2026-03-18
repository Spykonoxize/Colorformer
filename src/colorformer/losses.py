"""Loss functions used in Colorformer GAN training."""

from typing import Dict

import kornia
import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss computed on VGG19 feature maps."""

    def __init__(self, layer_idx: int = 16, device: str = "cuda") -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[: layer_idx + 1]
        self.vgg = vgg.to(device).eval()

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        pred_norm = self.normalize(pred_rgb)
        target_norm = self.normalize(target_rgb)

        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)

        return nn.functional.mse_loss(pred_features, target_features)


class ColorformerLoss(nn.Module):
    """Combined WGAN-GP + L1 + VGG loss."""

    def __init__(
        self,
        lambda_gan: float = 0.5,
        lambda_l1: float = 100.0,
        lambda_vgg: float = 1000.0,
        lambda_gp: float = 10.0,
        vgg_layer: int = 16,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.lambda_gan = lambda_gan
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_gp = lambda_gp
        self.device = device

        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGPerceptualLoss(layer_idx=vgg_layer, device=device)

    def gradient_penalty(
        self,
        discriminator: nn.Module,
        real_ab: torch.Tensor,
        fake_ab: torch.Tensor,
        l_condition: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_ab.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_ab)

        interpolated_ab = (alpha * real_ab + (1 - alpha) * fake_ab).requires_grad_(True)
        d_interpolated = discriminator(l_condition, interpolated_ab)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated_ab,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()

    def lab_to_rgb(self, l_channel: torch.Tensor, ab_channel: torch.Tensor) -> torch.Tensor:
        l_denorm = (l_channel + 1.0) * 50.0
        ab_denorm = ab_channel * 128.0

        lab = torch.cat([l_denorm, ab_denorm], dim=1)
        rgb = kornia.color.lab_to_rgb(lab)
        return rgb.clamp(0, 1)

    def forward_generator(
        self,
        fake_ab: torch.Tensor,
        real_ab: torch.Tensor,
        l_condition: torch.Tensor,
        discriminator: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        d_fake = discriminator(l_condition, fake_ab)
        loss_gan = -d_fake.mean()
        loss_l1 = self.l1_loss(fake_ab, real_ab)

        fake_rgb = self.lab_to_rgb(l_condition, fake_ab)
        real_rgb = self.lab_to_rgb(l_condition, real_ab)
        loss_vgg = self.vgg_loss(fake_rgb, real_rgb)

        loss_total = (
            self.lambda_gan * loss_gan
            + self.lambda_l1 * loss_l1
            + self.lambda_vgg * loss_vgg
        )

        return {
            "total": loss_total,
            "gan": loss_gan,
            "l1": loss_l1,
            "vgg": loss_vgg,
        }

    def forward_discriminator(
        self,
        fake_ab: torch.Tensor,
        real_ab: torch.Tensor,
        l_condition: torch.Tensor,
        discriminator: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        d_real = discriminator(l_condition, real_ab)
        d_fake = discriminator(l_condition, fake_ab.detach())

        loss_wgan = d_fake.mean() - d_real.mean()
        gradient_penalty = self.gradient_penalty(
            discriminator=discriminator,
            real_ab=real_ab,
            fake_ab=fake_ab.detach(),
            l_condition=l_condition,
        )

        loss_total = loss_wgan + self.lambda_gp * gradient_penalty

        return {
            "total": loss_total,
            "wgan": loss_wgan,
            "gp": gradient_penalty,
            "d_real": d_real.mean(),
            "d_fake": d_fake.mean(),
        }
