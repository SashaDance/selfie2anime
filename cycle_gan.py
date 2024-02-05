from discriminator import Discriminator
from generator import Generator

import torch.nn as nn
import torch

class CycleGAN:
    def __init__(self, device: torch.device,
                 in_channels: int = 3,
                 filters: int = 64,
                 num_resid: int = 6):
        # generators
        self.gen_XY = Generator(in_channels, filters, num_resid).to(device)
        self.gen_YX = Generator(in_channels, filters, num_resid).to(device)

        # discriminators
        self.dis_XY = Discriminator(in_channels, filters)
        self.dis_YX = Discriminator(in_channels, filters)


