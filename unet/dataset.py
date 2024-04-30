import torch
import torch.nn.functional as F

import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import VOCSegmentation as TorchDataset
from torchvision.transforms import ConvertImageDtype, Grayscale, RandomCrop, Resize, ToTensor

import os

from config import Config

class SegmentationDataset:
    def __init__(self, config: Config) -> None:
        self.batch_size = config.batch_size
        self.dataset_path = dataset_directory_path(config.dataset_folder_name)

    def retrieve_train_data(self) -> Dataset:
        return self._create_dataloader(image_set='train', shuffle=True)

    def retrieve_val_data(self) -> Dataset:
        return self._create_dataloader(image_set='trainval', shuffle=False)

    def retrieve_test_data(self) -> Dataset:
        return self._create_dataloader(image_set='val', shuffle=False)

    def _create_dataloader(self, image_set: str, shuffle: bool) -> DataLoader:
        dataset = self._create_dataset(image_set)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
        )

        return dataloader

    def _create_dataset(self, image_set: str) -> Dataset:
        return TorchDataset(
            root=self.dataset_path,
            image_set=image_set,
            download=True,
            transforms=Transforms()
        )

class Transforms():
    def __init__(self):
        self.common_transforms = []

        self.image_transforms = [
            Grayscale(),
            ToTensor(),
            Resize(size=(572, 572), antialias=True)
        ]

        self.segmentation_transforms = [
            ToTensorSegmentation(),
            Resize(size=(388, 388), antialias=True)
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
        x[x == 255] = 23

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


def compute_class_weights(dataset):
    class_weights = torch.zeros(24)

    train_balance_dataset = dataset.retrieve_train_data()
    for _, y in train_balance_dataset:
        total = y.numel()
        for label in y.unique():
            label = label.item()
            if label not in class_weights:
                class_weights[label] = 0
            class_weights[label] += (y == label).sum() / total

    class_weights = 1 / (class_weights + 1e-7)
    class_weights /= class_weights.sum()
    del train_balance_dataset
    return class_weights


if __name__ == '__main__':
    from config import get_config

    # Sanity check if script run manually
    config = get_config()
    for x in SegmentationDataset(config).get_train_dataset():
        print(x[0].shape, x[1].shape)
        exit()
