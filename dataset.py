import config

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import os


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
