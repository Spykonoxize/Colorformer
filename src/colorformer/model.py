"""Model architecture for Colorformer GAN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFFN(nn.Module):
    """Color Feed-Forward Network used inside ColorFormer blocks."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc2(x)
        return x


class WindowPartition:
    """Helpers to split/reconstruct tensors into local windows."""

    @staticmethod
    def partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
        bsz, height, width, channels = x.shape
        x = x.view(
            bsz,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        )
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, window_size, window_size, channels
        )
        return windows

    @staticmethod
    def reverse(
        windows: torch.Tensor,
        window_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        bsz = int(windows.shape[0] / (height * width / window_size / window_size))
        x = windows.view(
            bsz,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bsz, height, width, -1)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords = torch.stack(
            torch.meshgrid(
                [torch.arange(window_size), torch.arange(window_size)], indexing="ij"
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz_windows, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bsz_windows, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(bsz_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz_windows, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LWMHSA(nn.Module):
    """Local window multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def calculate_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        img_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        count = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = count
                count += 1

        mask_windows = WindowPartition.partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, height, width, channels = x.shape

        pad_left = pad_top = 0
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
        _, padded_h, padded_w, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.calculate_mask(padded_h, padded_w, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = WindowPartition.partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)

        shifted_x = WindowPartition.reverse(attn_windows, self.window_size, padded_h, padded_w)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_right > 0 or pad_bottom > 0:
            x = x[:, :height, :width, :].contiguous()

        return x


class ColorFormerBlock(nn.Module):
    """Core ColorFormer residual block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LWMHSA(dim, num_heads, window_size, shift_size)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = CFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dwconv(x)

        # Attention blocks operate on NHWC layout.
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        x_nhwc = x_nhwc + self.attn(self.norm1(x_nhwc))
        x_nhwc = x_nhwc + self.ffn(self.norm2(x_nhwc))

        return x_nhwc.permute(0, 3, 1, 2).contiguous()


class ColorFormerUNet(nn.Module):
    """U-Net style generator based on ColorFormer blocks."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 64,
        num_heads: list[int] | tuple[int, int, int, int] = (4, 8, 16, 32),
        window_size: int = 7,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc1 = ColorFormerBlock(base_channels, num_heads[0], window_size)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)

        self.enc2 = ColorFormerBlock(base_channels * 2, num_heads[1], window_size)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        self.enc3 = ColorFormerBlock(base_channels * 4, num_heads[2], window_size)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        self.enc4 = ColorFormerBlock(base_channels * 8, num_heads[3], window_size)
        self.down4 = nn.Conv2d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1)

        self.bottleneck = ColorFormerBlock(base_channels * 8, num_heads[3], window_size)

        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, stride=2)
        self.dec1 = ColorFormerBlock(base_channels * 16, num_heads[3], window_size)
        self.proj1 = nn.Conv2d(base_channels * 16, base_channels * 4, 1)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)
        self.dec2 = ColorFormerBlock(base_channels * 8, num_heads[2], window_size)
        self.proj2 = nn.Conv2d(base_channels * 8, base_channels * 2, 1)

        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.dec3 = ColorFormerBlock(base_channels * 4, num_heads[1], window_size)
        self.proj3 = nn.Conv2d(base_channels * 4, base_channels, 1)

        self.up4 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec4 = ColorFormerBlock(base_channels * 2, num_heads[0], window_size)

        self.output_proj = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_proj(x)

        e1 = self.enc1(x1)
        e1_res = x1 + e1
        d1 = self.down1(e1_res)

        e2 = self.enc2(d1)
        e2_res = d1 + e2
        d2 = self.down2(e2_res)

        e3 = self.enc3(d2)
        e3_res = d2 + e3
        d3 = self.down3(e3_res)

        e4 = self.enc4(d3)
        e4_res = d3 + e4
        d4 = self.down4(e4_res)

        b = self.bottleneck(d4)

        u1 = self.up1(b)
        u1_cat = torch.cat([u1, e4_res], dim=1)
        dec1 = self.dec1(u1_cat)
        dec1 = self.proj1(dec1)

        u2 = self.up2(dec1)
        u2_cat = torch.cat([u2, e3_res], dim=1)
        dec2 = self.dec2(u2_cat)
        dec2 = self.proj2(dec2)

        u3 = self.up3(dec2)
        u3_cat = torch.cat([u3, e2_res], dim=1)
        dec3 = self.dec3(u3_cat)
        dec3 = self.proj3(dec3)

        u4 = self.up4(dec3)
        u4_cat = torch.cat([u4, e1_res], dim=1)
        dec4 = self.dec4(u4_cat)

        return self.output_proj(dec4)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator conditioned on L channel and predicted ab."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(
                    base_channels * nf_mult_prev,
                    base_channels * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(base_channels * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(
                base_channels * nf_mult_prev,
                base_channels * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, l_channel: torch.Tensor, ab_channel: torch.Tensor) -> torch.Tensor:
        x = torch.cat([l_channel, ab_channel], dim=1)
        return self.model(x)
