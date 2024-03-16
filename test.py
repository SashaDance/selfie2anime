from src.model.dataset import ImageDataset, process_img_to_show, ImageBuffer
from src.model.generator import Generator
from src.model.discriminator import Discriminator
from src.model.cycle_gan import CycleGAN
import src.model.config as config

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
torch.manual_seed(42)
def test_image_dataset(in_channels: int = 3, image_size: int = 128) -> None:
    female_tr = ImageDataset('trainA')
    anime_tr = ImageDataset('trainB')
    female_test = ImageDataset('testA')
    anime_test = ImageDataset('testB')

    assert len(female_tr) == len(anime_tr) == 3400, \
        f'Lost some train data: only {len(female_tr)} and {len(anime_tr)} of 3400'

    assert len(female_test) == len(anime_test) == 100, \
        f'Lost some train data: only {len(female_test)} and {len(anime_test)} of 100'

    img_shape = [in_channels, image_size, image_size]
    assert list(female_tr[0].shape) == img_shape, \
        f'Incorrect image shape: got {list(female_tr[0].shape)}, expected: {img_shape}'
    assert anime_tr[0].shape == torch.Size([3, 128, 128]), \
        f'Incorrect image shape: got {list(anime_tr[0].shape)}, expected: {img_shape}'
    assert female_test[0].shape == torch.Size([3, 128, 128]), \
        f'Incorrect image shape: got {list(female_test[0].shape)}, expected: {img_shape}'
    assert anime_test[0].shape == torch.Size([3, 128, 128]), \
        f'Incorrect image shape: got {list(anime_test[0].shape)}, expected: {img_shape}'


def test_generator(in_channels: int = 3, image_size: int = 128,
                   batch_size: int = 3) -> None:
    x = torch.randn(size=[batch_size, in_channels, image_size, image_size])
    generator = Generator(in_channels=in_channels)

    output = generator(x)
    assert list(x.shape) == list(output.shape), \
        f'Input and output shapes should be the same: got {list(output.shape)}, expected {list(x.shape)}'


def test_discriminator(in_channels: int = 3, image_size: int = 128,
                       batch_size: int = 1) -> None:
    x = torch.randn(size=[batch_size, in_channels, image_size, image_size])
    discriminator = Discriminator(in_channels=in_channels)

    output = discriminator(x)
    # print(output.shape)
    assert list(output.shape)[1] == 1, \
        f'Incorrect output shape: output must have 1 channel, got: {list(output.shape)[1]}'


def test_train_loop(left_ind: int = 0,
                    right_ind: int = 5,
                    epochs: int = 2) -> None:
    female_tr = ImageDataset('trainA')
    anime_tr = ImageDataset('trainB')
    female_test = ImageDataset('testA')
    anime_test = ImageDataset('testB')

    loader_x = DataLoader(female_tr[left_ind: right_ind])
    loader_y = DataLoader(anime_tr[left_ind: right_ind])

    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )

    model = CycleGAN(device)

    discriminator_params_x = list(model.dis_X.parameters())
    discriminator_params_y = list(model.dis_Y.parameters())
    generator_params = (
            list(model.gen_XY.parameters()) + list(model.gen_YX.parameters())
    )
    optimizers = {
        'discriminator_x': torch.optim.Adam(
            discriminator_params_x, lr=config.LEARNING_RATE, betas=config.BETAS
        ),
        'discriminator_y': torch.optim.Adam(
            discriminator_params_y, lr=config.LEARNING_RATE, betas=config.BETAS
        ),
        'generator': torch.optim.Adam(
            generator_params, lr=config.LEARNING_RATE, betas=config.BETAS
        )
    }

    losses = model.train(
        epochs, 1, optimizers, loader_x, loader_y, female_test, anime_test
    )


def test_generation(model_path: str,
                    left_ind: int = 0,
                    right_ind: int = 5) -> None:
    model = Generator()
    model.load_state_dict(torch.load(
        model_path,
        map_location=torch.device('cpu')
    ))
    model.eval()

    test_images = ImageDataset('testA')
    images = test_images[left_ind: right_ind]

    fig, ax = plt.subplots(len(images), 2)
    for i in range(len(images)):
        image = process_img_to_show(images[i])
        output = process_img_to_show(model(images[i]))

        ax[i][0].imshow(image)
        ax[i][0].axis('off')
        ax[i][1].imshow(output)
        ax[i][1].axis('off')

    plt.show()

    # TODO: add moving images to device on inference


def test_image_buffer(batch_size: int = 5):
    dataloader = DataLoader(ImageDataset('trainA')[0:20],
                            batch_size=batch_size)
    buffer = ImageBuffer(buffer_lim=10)
    model = Generator()
    model.load_state_dict(torch.load(
        'model_checkpoints/epoch_20/gen_XY(3)',
        map_location=torch.device('cpu')
    ))
    for image_batch in dataloader:
        fig, ax = plt.subplots(batch_size, 2)
        images = model(image_batch)
        images = buffer.get_images(images)
        for i in range(batch_size):
            image_1 = process_img_to_show(images[i])
            image_2 = process_img_to_show(image_batch[i])
            ax[i][0].imshow(image_1)
            ax[i][0].axis('off')
            ax[i][1].imshow(image_2)
            ax[i][1].axis('off')

        plt.show()

    fig, ax = plt.subplots(len(buffer.buffer), 1)
    for i, image in enumerate(buffer.buffer):
        image = process_img_to_show(image)
        ax[i].imshow(image)
        ax[i].axis('off')

    plt.show()


def main() -> None:
    test_image_dataset()
    test_generator()
    test_discriminator()
    test_train_loop()
    # test_generation('model_checkpoints/epoch_20/gen_XY(3)')
    # test_image_buffer()


if __name__ == '__main__':
    main()
