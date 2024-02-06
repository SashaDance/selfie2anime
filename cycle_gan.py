from discriminator import Discriminator
from generator import Generator

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn as nn
import torch


class CycleGAN:
    """
    CycleGAN architecture consists of two Generators and two Discriminators
    """
    def __init__(self, device: torch.device,
                 in_channels: int = 3,
                 filters: int = 64,
                 num_resid: int = 6):
        self.device = device

        # generator from X to Y
        self.gen_XY = Generator(in_channels, filters, num_resid).to(device)
        # generator from X to Y
        self.gen_YX = Generator(in_channels, filters, num_resid).to(device)

        # discriminator of X images
        self.dis_X = Discriminator(in_channels, filters)
        # discriminator of Y images
        self.dis_Y = Discriminator(in_channels, filters)

    def __discriminator_step(self, optimizer: Optimizer,
                             x_batch: torch.Tensor,
                             y_batch: torch.Tensor) -> torch.Tensor:
        optimizer.zero_grad()

        # discriminator X

        # teaching discriminator to detect real images
        real_x_preds = self.dis_X(x_batch)

        real_x_loss = torch.mean(
            (real_x_preds - 1) ** 2  # 1 is for the real images
        )

        # teaching discriminator to detect fake images
        fake_x = self.gen_XY(x_batch)
        fake_x_preds = self.dis_X(fake_x)

        fake_x_loss = torch.mean(
            (fake_x_preds - 0) ** 2  # 0 is for the real images
        )

        # updating weights for discriminator
        loss_x = real_x_loss + fake_x_loss
        loss_x.backward()
        optimizer.step()

        # discriminator Y

        # teaching discriminator to detect real images
        real_y_preds = self.dis_Y(y_batch)
        real_y_loss = torch.mean(
            (real_y_preds - 1) ** 2  # 1 is for the real images
        )

        # teaching discriminator to detect fake images
        fake_y = self.gen_YX(y_batch)
        fake_y_preds = self.dis_Y(fake_y)

        fake_y_loss = torch.mean(
            (fake_y_preds - 0) ** 2  # 0 is for the real images
        )

        # updating weights for discriminator
        loss_y = real_y_loss + fake_y_loss
        loss_y.backward()
        optimizer.step()

        return torch.Tensor()

    def __generator_step(self, optimizer: Optimizer,
                         x_batch: torch.Tensor,
                         y_batch: torch.Tensor) -> torch.Tensor:
        optimizer.zero_grad()

        # generator from X to Y

        # teaching generator to 'fool' discriminator
        fake_x = self.gen_XY(x_batch)


        return torch.Tensor()

    def train(self, epochs: int,
              optimizers: dict[str, Optimizer],
              train_loader_x: DataLoader,
              train_loader_y: DataLoader) -> None:
        losses = {}

        for epoch in range(epochs):
            for x_batch, y_batch in tqdm(zip(train_loader_y, train_loader_x)):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.__discriminator_step(
                    optimizers['discriminator'], x_batch, y_batch
                )
                self.__generator_step(
                    optimizers['generator'], x_batch, y_batch
                )

                # TODO: save losses
