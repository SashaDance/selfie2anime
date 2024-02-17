import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, padding=1,
                stride=stride, **kwargs
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    """
    Discriminator of CycleGAN model,
    architecture is the same as in the original paper
    """

    def __init__(self, in_channels: int = 3, filters: int = 64):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels, filters, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2)
        )
        self.conv1 = ConvLayer(
            filters, filters * 2, stride=2,
        )
        self.conv2 = ConvLayer(
            filters * 2, filters * 4, stride=2,
        )
        self.conv3 = ConvLayer(
            filters * 4, filters * 8, stride=1
        )

        self.last = nn.Sequential(
            nn.Conv2d(
                filters * 8, 1, kernel_size=4, stride=1, padding=1
            ),
            nn.Sigmoid()  # we need a probabilities for each patch
        )

    def forward(self, x):
        # convolutions
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # probabilities
        x = self.last(x)

        return x
