from discriminator import Discriminator
from generator import Generator
from dataset import ImageDataset, process_img_to_show, ImageBuffer
import config

import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch
import numpy as np
import matplotlib.pyplot as plt


class CycleGAN:
    """
    CycleGAN architecture consists of two Generators and two Discriminators
    """
    def __init__(self, device: torch.device,
                 in_channels: int = 3,
                 filters: int = 64,
                 num_resid: int = 6,
                 use_buffer: bool = True,
                 init_dir: str = None):
        self.device = device

        # generator from X to Y
        self.gen_XY = Generator(in_channels, filters, num_resid).to(device)
        # generator from Y to X
        self.gen_YX = Generator(in_channels, filters, num_resid).to(device)

        # discriminator X
        self.dis_X = Discriminator(in_channels, filters).to(device)
        # discriminator Y
        self.dis_Y = Discriminator(in_channels, filters).to(device)

        self.use_buffer = use_buffer
        if use_buffer:
            self.generated_x_buffer = ImageBuffer(self.device)
            self.generated_y_buffer = ImageBuffer(self.device)

        if init_dir:
            self.gen_XY.load_state_dict(
                torch.load(os.path.join(init_dir, 'gen_XY'))
            )
            self.gen_YX.load_state_dict(
                torch.load(os.path.join(init_dir, 'gen_YX'))
            )
            self.dis_X.load_state_dict(
                torch.load(os.path.join(init_dir, 'dis_X'))
            )
            self.dis_Y.load_state_dict(
                torch.load(os.path.join(init_dir, 'dis_Y'))
            )

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

    def __discriminator_step(self, optimizers: dict[str, Optimizer],
                             x_batch: torch.Tensor,
                             y_batch: torch.Tensor) -> list[float, float]:
        """
        :param optimizers: optimizers with weights from both discriminators
        :param x_batch:
        :param y_batch:
        :return: losses
        """
        # discriminator X
        optimizers['discriminator_x'].zero_grad()

        # teaching discriminator to detect real (from X) images
        real_x_preds = self.dis_X(x_batch)

        real_x_loss = torch.mean(
            (real_x_preds - 1) ** 2  # 1 is for the real (from X) images
        )

        # teaching discriminator to detect fake (not from X) images
        generated_x = self.gen_YX(y_batch)
        if self.use_buffer:
            generated_x = self.generated_x_buffer.get_images(generated_x)

        generated_x_preds = self.dis_X(generated_x)

        generated_x_loss = torch.mean(
            (generated_x_preds - 0) ** 2  # 0 is for the fake (not from X) images
        )

        # updating weights for discriminator
        loss_x = real_x_loss + generated_x_loss
        loss_x.backward()
        optimizers['discriminator_x'].step()

        # discriminator Y
        optimizers['discriminator_y'].zero_grad()

        # teaching Y discriminator to detect real (from Y) images
        real_y_preds = self.dis_Y(y_batch)
        real_y_loss = torch.mean(
            (real_y_preds - 1) ** 2  # 1 is for the real (from Y) images
        )

        # teaching discriminator to detect fake (not from Y) images
        generated_y = self.gen_XY(x_batch)
        if self.use_buffer:
            generated_y = self.generated_y_buffer.get_images(generated_y)

        generated_y_preds = self.dis_Y(generated_y)

        generated_y_loss = torch.mean(
            (generated_y_preds - 0) ** 2  # 0 is for the fake (not from Y) images
        )

        # updating weights for discriminator
        loss_y = real_y_loss + generated_y_loss
        loss_y.backward()
        optimizers['discriminator_y'].step()

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
        generated_y = self.gen_XY(x_batch)
        generated_y_preds = self.dis_Y(generated_y)

        generated_y_loss = torch.mean(
            (generated_y_preds - 1) ** 2  # 1 is for the real (from Y) images
        )

        # calculating cycle consistency loss
        consistency_loss_x = self.cycle_consistency_loss(
            self.gen_YX(generated_y), x_batch
        )

        # generator from Y to X

        # teaching generator to 'fool' discriminator
        generated_x = self.gen_YX(y_batch)
        generated_x_preds = self.dis_X(generated_x)

        generated_x_loss = torch.mean(
            (generated_x_preds - 1) ** 2  # 1 is for the real (from X) images
        )

        # calculating cycle consistency loss
        consistency_loss_y = self.cycle_consistency_loss(
            self.gen_XY(generated_x), y_batch
        )

        loss = (
            generated_y_loss
            + generated_x_loss
            + config.LAMBDA * consistency_loss_x
            + config.LAMBDA * consistency_loss_y
        )

        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, epochs: int,
              save_rate: int,
              optimizers: dict[str, Optimizer],
              train_loader_x: DataLoader,
              train_loader_y: DataLoader,
              test_dataset_x: ImageDataset,
              test_dataset_y: ImageDataset,
              show_images: bool = True) -> dict[str, list]:
        """
        :param epochs:
        :param save_rate: save model every save_rate epochs
        :param optimizers:
        :param train_loader_x:
        :param train_loader_y:
        :param test_dataset_x:
        :param test_dataset_y:
        :param show_images: show current results or not
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
            for x_batch, y_batch in tqdm(zip(train_loader_x, train_loader_y)):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # updating weights and calculating losses
                loss_gen_batch = self.__generator_step(
                    optimizers['generator'], x_batch, y_batch
                )
                loss_x_d_batch, loss_y_d_batch = self.__discriminator_step(
                    optimizers, x_batch, y_batch
                )

                loss_x_d += loss_x_d_batch
                loss_y_d += loss_y_d_batch
                loss_gen += loss_gen_batch

            # showing the images
            if show_images:
                ind_x = np.random.randint(low=0, high=len(test_dataset_x))
                ind_y = np.random.randint(low=0, high=len(test_dataset_y))

                image_x = test_dataset_x[ind_x]
                image_y = test_dataset_y[ind_y]
                generated_y = self.gen_XY(image_x.to(self.device))
                generated_x = self.gen_YX(image_y.to(self.device))

                fig, ax = plt.subplots(2, 2)
                ax[0][0].imshow(process_img_to_show(image_x))
                ax[0][0].axis('off')
                ax[0][1].imshow(process_img_to_show(generated_y))
                ax[0][1].axis('off')
                ax[1][0].imshow(process_img_to_show(image_y))
                ax[1][0].axis('off')
                ax[1][1].imshow(process_img_to_show(generated_x))
                ax[1][1].axis('off')

                plt.show()

            # saving model checkpoint
            if epoch % save_rate == 0:
                save_dir = os.path.join(config.SAVE_PATH, f'epoch_{epoch}')
                try:
                    os.mkdir(save_dir)
                    torch.save(
                        self.dis_X.state_dict(),
                        os.path.join(save_dir, 'dis_X')
                    )
                    torch.save(
                        self.dis_Y.state_dict(),
                        os.path.join(save_dir, 'dis_Y')
                    )
                    torch.save(
                        self.gen_XY.state_dict(),
                        os.path.join(save_dir, 'gen_XY')
                    )
                    torch.save(
                        self.gen_YX.state_dict(),
                        os.path.join(save_dir, 'gen_YX')
                    )
                except FileExistsError:
                    print(
                        f'Unable to save checkpoint on epoch {epoch}: ',
                        f'dir {save_dir} already exists'
                    )

            loss_x_d = loss_x_d / len(train_loader_y)
            loss_y_d = loss_y_d / len(train_loader_y)
            loss_gen = loss_gen / len(train_loader_y)
            # printing losses
            print(
                f'Epoch - {epoch} | dis_x_loss: {loss_x_d}, ',
                f'dis_y_loss: {loss_y_d} | gen_loss: {loss_gen}'
            )
            # saving losses
            losses['loss_x_dis'].append(loss_x_d)
            losses['loss_y_dis'].append(loss_y_d)
            losses['loss_gen'].append(loss_gen)

        return losses

        # TODO: add saving model when val loss is min
        # TODO: implement training loop for loaders with different sizes
        # TODO: add calculating val losses
