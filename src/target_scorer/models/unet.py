"""
High-Resolution Feature Pyramid U-Net architecture for sub-pixel heatmap regression.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Standardized Convolution-BatchNorm-Mish block with residual skip."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.residual = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.Mish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x) + self.residual(x))


class HighResHeatmapUNet(nn.Module):
    """
    High-Resolution Feature Pyramid U-Net architecture for continuous sub-pixel regression.
    Outputs: Channel 0 = Bullet Impacts, Channel 1 = Bullseye Center.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 2, base_dim: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.Mish(inplace=True),
            ConvBlock(base_dim, base_dim),
        )
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_dim, base_dim * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_dim * 2, base_dim * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_dim * 4, base_dim * 8))

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ConvBlock(base_dim * 8 + base_dim * 4, base_dim * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ConvBlock(base_dim * 4 + base_dim * 2, base_dim * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(base_dim * 2 + base_dim, base_dim)

        self.final_head = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.Mish(inplace=True),
            nn.Conv2d(base_dim, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 8 != 0 or x.shape[-1] % 8 != 0:
            raise ValueError(
                f"Input spatial dimensions (H={x.shape[-2]}, W={x.shape[-1]}) "
                "must be divisible by 8 for U-Net down/up-sampling architecture."
            )
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        d2 = self.dec3(torch.cat([self.up3(s3), s2], dim=1))
        d1 = self.dec2(torch.cat([self.up2(d2), s1], dim=1))
        d0 = self.dec1(torch.cat([self.up1(d1), s0], dim=1))
        return self.final_head(d0)
