
import torch.nn as nn


class DownConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, activation: bool = True, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding_mode='reflect',**kwargs
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if activation else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class UpConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, activation: bool = True, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, **kwargs),
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


class Generator(nn.Module):
    """
    Generator of CycleGAN model,
    architecture is the same as in the original paper
    """
    def __init__(self, in_channels: int = 3,
                 filters: int = 64,
                 num_resid: int = 6):
        super().__init__()
        self.down_conv0 = DownConvLayer(
            in_channels, filters, 7, padding=3, stride=1
        )
        self.down_conv1 = DownConvLayer(
            filters, filters * 2, 3, padding=1, stride=2
        )
        self.down_conv2 = DownConvLayer(
            filters * 2, filters * 4, 3, padding=1, stride=2
        )

        self.resid_blocks = nn.Sequential(
            *[ResidualBlock(filters * 4) for _ in range(num_resid)]
        )

        self.up_conv0 = UpConvLayer(
            filters * 4, filters * 2, 3, padding=1, stride=2, output_padding=1
        )
        self.up_conv1 = UpConvLayer(
            filters * 2, filters, 3, padding=1, stride=2, output_padding=1
        )

        self.last = nn.Sequential(
            nn.Conv2d(
                filters, in_channels, 7, stride=1,
                padding=3, padding_mode='reflect'
            ),
            nn.Tanh()
        )

    def forward(self, x):
        # downsampling
        x = self.down_conv0(x)
        x = self.down_conv1(x)
        x = self.down_conv2(x)

        # residual blocks
        x = self.resid_blocks(x)

        # upsampling
        x = self.up_conv0(x)
        x = self.up_conv1(x)

        x = self.last(x)

        return x