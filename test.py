from dataset import ImageDataset
import torch


def test_image_dataset(image_size: int = 128) -> None:
    female_tr = ImageDataset('trainA')
    anime_tr = ImageDataset('trainB')
    female_test = ImageDataset('testA')
    anime_test = ImageDataset('testB')

    assert len(female_tr) == len(anime_tr) == 3400, \
        f'Lost some train data: only {len(female_tr)} and {len(anime_tr)} of 3400'

    assert len(female_test) == len(anime_test) == 100, \
        f'Lost some train data: only {len(female_test)} and {len(anime_test)} of 100'

    print(female_tr[0].shape)
    print(anime_tr[1].shape)
    print(female_test[2].shape)
    print(anime_test[3].shape)


def main() -> None:
    test_image_dataset()


if __name__ == '__main__':
    main()
