import torch
import torch.nn.functional as F

import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import VOCSegmentation as TorchDataset
from torchvision.transforms import ConvertImageDtype, Grayscale, RandomCrop, Resize, ToTensor

import os

from config import Config

# TODO: Do data augmentation
class SegmentationDataset:
    def __init__(self, config: Config) -> None:
        self.batch_size = config.batch_size
        self.dataset_path = dataset_directory_path(config.dataset_folder_name)

    def get_train_dataset(self) -> Dataset:
        return self._create_dataloader(image_set='train', shuffle=True)

    def get_val_dataset(self) -> Dataset:
        return self._create_dataloader(image_set='val', shuffle=False)

    def get_test_dataset(self) -> Dataset:
        return self._create_dataloader(image_set='test', shuffle=False)

    def _create_dataloader(self, image_set: str, shuffle: bool) -> DataLoader:
        dataset = self._create_dataset(image_set)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            # num_workers=1,
        )

        return dataloader

    def _create_dataset(self, image_set: str) -> Dataset:
        return TorchDataset(
            root=self.dataset_path,
            image_set=image_set,
            download=True,
            transforms=Transforms()
        )

# TODO: Shouldn't output be classes
class Transforms():
    def __init__(self):
        self.common_transforms = [
            RandomCrop(size=(100, 100)), # TODO: Don't hard-code this
        ]

        self.image_transforms = [
            Grayscale(),
            ToTensor(),
            Resize(size=(572, 572), antialias=True) # TODO: Don't hard-code this!!!
        ]

        self.segmentation_transforms = [
            ToTensorSegmentation(),
            Resize(size=(388, 388), antialias=True) # TODO: Don't hard-code this!!!
        ]


    def __call__(self, image, segmentation) -> Tensor:
        for transform in self.common_transforms:
            image = transform(image)
            segmentation = transform(segmentation)
            

        for transform in self.image_transforms:
            image = transform(image)

        for transform in self.segmentation_transforms:
            segmentation = transform(segmentation)

        return image, segmentation


class ToTensorSegmentation():
    def __call__(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(np.array(x), dtype=torch.int64)
        x = x.unsqueeze(0)

        # Remove border class
        x[x == 255] = 0

        return x
        

# Helpers
def create_directory_if_does_not_exist(directory: str) -> None:
    if os.path.exists(directory):
        return

    os.mkdir(directory)

def dataset_directory_path(dataset_folder_name: str) -> str:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    root_directory = os.path.dirname(current_directory)
    dataset_directory = os.path.join(root_directory, dataset_folder_name)
    return dataset_directory

if __name__ == '__main__':
    from config import get_config

    # Sanity check if script run manually
    config = get_config()
    for x in SegmentationDataset(config).get_train_dataset():
        print(x[0].shape, x[1].shape)
        exit()
