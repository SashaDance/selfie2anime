from discriminator import Discriminator
from generator import Generator
import config

import os
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
        self.dis_X = Discriminator(in_channels, filters).to(device)
        # discriminator of Y images
        self.dis_Y = Discriminator(in_channels, filters).to(device)

    @staticmethod
    def cycle_consistency_loss(reconstructed: torch.Tensor,
                                real: torch.Tensor) -> torch.Tensor:
        """
        We want mapping (generators) F from X to Y and G from X to Y
        to be inverses to each other: F^(-1) = G and G^(-1) = F

        :param reconstructed: gen_YX((gen_XY(x))) for x images
                              and gen_XY((gen_YX(y))) for y images
        :param real: real image
        :return: L1 loss between real and reconstructed images
        """

        return torch.mean(torch.abs(real - reconstructed))

    def __discriminator_step(self, optimizer: Optimizer,
                             x_batch: torch.Tensor,
                             y_batch: torch.Tensor) -> list[float, float]:
        """
        :param optimizer: optimizer with weights from both discriminators
        :param x_batch:
        :param y_batch:
        :return: losses
        """
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
            (fake_x_preds - 0) ** 2  # 0 is for the fake images
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
            (fake_y_preds - 0) ** 2  # 0 is for the fake images
        )

        # updating weights for discriminator
        loss_y = real_y_loss + fake_y_loss
        loss_y.backward()
        optimizer.step()

        return [loss_x.item(), loss_y.item()]

    def __generator_step(self, optimizer: Optimizer,
                         x_batch: torch.Tensor,
                         y_batch: torch.Tensor) -> float:
        """
        :param optimizer: optimizer with weights from both generators
        :param x_batch:
        :param y_batch:
        :return: losses
        """
        optimizer.zero_grad()

        # generator from X to Y

        # teaching generator to 'fool' discriminator
        fake_x = self.gen_XY(x_batch)
        fake_x_preds = self.dis_X(fake_x)

        fake_x_loss = torch.mean(
            (fake_x_preds - 1) ** 2  # 1 is for the real images
        )

        # calculating cycle consistency loss
        consistency_loss_x = self.cycle_consistency_loss(
            self.gen_YX(fake_x), x_batch
        )

        # generator from Y to X

        # teaching generator to 'fool' discriminator
        fake_y = self.gen_YX(y_batch)
        fake_y_preds = self.dis_Y(fake_y)

        fake_y_loss = torch.mean(
            (fake_y_preds - 1) ** 2  # 1 is for the real images
        )

        # calculating cycle consistency loss
        consistency_loss_y = self.cycle_consistency_loss(
            self.gen_XY(fake_y), y_batch
        )

        loss = (
            fake_x_loss
            + fake_y_loss
            + consistency_loss_x
            + consistency_loss_y
        )

        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, epochs: int,
              save_rate: int,
              optimizers: dict[str, Optimizer],
              train_loader_x: DataLoader,
              train_loader_y: DataLoader,) -> dict[str, list]:
        """
        :param epochs:
        :param save_rate: save model every save_rate epochs
        :param optimizers:
        :param train_loader_x:
        :param train_loader_y:
        :return: dict with losses
        """

        losses = {
            'loss_x_dis': [],
            'loss_y_dis': [],
            'loss_gen': []
        }
        assert len(train_loader_x) == len(train_loader_y), \
            f'Loaders of x and y should be the same size'

        for epoch in range(epochs):
            loss_x_d = 0
            loss_y_d = 0
            loss_gen = 0
            for x_batch, y_batch in tqdm(zip(train_loader_y, train_loader_x)):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # updating weights and calculating losses
                loss_x_d_batch, loss_y_d_batch = self.__discriminator_step(
                    optimizers['discriminator'], x_batch, y_batch
                )
                loss_gen_batch = self.__generator_step(
                    optimizers['generator'], x_batch, y_batch
                )

                loss_x_d += loss_x_d_batch
                loss_y_d += loss_y_d_batch
                loss_gen += loss_gen_batch

            # saving model checkpoint
            if epoch % save_rate == 0:
                os.mkdir(os.path.join(config.SAVE_PATH, f'epoch_{epoch}'))
                torch.save(
                    self.dis_X.state_dict(),
                    os.path.join(config.SAVE_PATH, f'epoch_{epoch}/dis_X')
                )
                torch.save(
                    self.dis_Y.state_dict(),
                    os.path.join(config.SAVE_PATH, f'epoch_{epoch}/dis_Y')
                )
                torch.save(
                    self.gen_XY.state_dict(),
                    os.path.join(config.SAVE_PATH, f'epoch_{epoch}/gen_XY')
                )
                torch.save(
                    self.gen_YX.state_dict(),
                    os.path.join(config.SAVE_PATH, f'epoch_{epoch}/gen_YX')
                )

            # saving losses
            losses['loss_x_dis'].append(loss_x_d / len(train_loader_y))
            losses['loss_y_dis'].append(loss_y_d / len(train_loader_y))
            losses['loss_gen'].append(loss_gen / len(train_loader_y))

        return losses

        # TODO: add printing number of epoch
        # TODO: add handling exceptions when saving the model
        # TODO: implement training loop for loaders with different sizes
        # TODO: add calculating val losses
        # TODO: implement buffer
