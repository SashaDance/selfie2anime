from dataset import ImageDataset
import torch


def test_image_dataset(in_channels:int = 3, image_size: int = 128) -> None:
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


def main() -> None:
    test_image_dataset()


if __name__ == '__main__':
    main()
