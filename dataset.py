import config

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import os
import numpy as np


def process_img_to_show(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().numpy()
    # scale from 0 to 255
    image = ((image - image.min()) * 255) / (image.max() - image.min())
    image = image.transpose(1, 2, 0).astype(np.uint8)

    return image


class ImageDataset(Dataset):
    def __init__(self, images_path: str,
                 in_channels: int = 3,
                 image_size: int = 128) -> None:
        # Getting list of paths to images
        images_dir = os.path.join(config.BASE_PATH, images_path)
        dir_list = os.listdir(images_dir)
        self.images_path = [
            os.path.join(images_dir, image) for image in dir_list
        ]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * in_channels,
                                 std=[0.5] * in_channels)
        ])

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[ind] for ind in range(key.start, key.stop)]
        image_path = self.images_path[key]
        image = Image.open(image_path).convert('RGB')

        image_transformed = self.transform(image)

        return image_transformed


class ImageBuffer:
    """
    Buffer of generated images, which gives generator some time
    to learn to fool the discriminator
    """

    def __init__(self, buffer_lim: int = 64, prob_threshold: float = 0.5):
        """
        :param buffer_lim: max size of buffer
        :param prob_threshold: probability of putting new image to buffer
        """
        assert config.BATCH_SIZE <= buffer_lim, \
            'Buffer limit should be greater than the batch size'
        self.buffer_lim = buffer_lim
        self.prob_threshold = prob_threshold
        self.buffer: list[torch.Tensor] = []

    def get_image(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        :param image_batch: current generated image barch
        :return: image batch after sampling from buffer
        """
        # return current batch if there is less images in the buffer than the batch size
        if len(self.buffer) < len(image_batch):
            return image_batch

        for image in image_batch:
            # initializing of buffer
            if len(self.buffer) < self.buffer_lim:
                self.buffer.append(image)
                images.append(image)
            # updating the buffer
            else:
                prob = np.random.uniform()
                ind = np.random.randint(0, self.buffer_lim)
                if prob > self.prob_threshold:
                    # add replace image in buffer


