"""
Generator of CycleGAN model,
architecture is the same as in the original paper
"""
import torch
import torch.nn as nn


class DownConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, activation: bool = True, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, padding_mode='reflect', **kwargs
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if activation else nn.Identity()
        )


class UpConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, activation: bool = True, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if activation else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resid = nn.Sequential(
            DownConvLayer(in_channels, in_channels, 3, padding='same'),
            DownConvLayer(
                in_channels, in_channels, 3, activation=False, padding='same'
            )
        )

    def forward(self, x):
        return x + self.resid(x)  # skip connection