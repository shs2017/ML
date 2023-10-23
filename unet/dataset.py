from torchvision.datasets import VOCSegmentation as TorchDataset

import os

DATASETS_FOLDER_NAME = 'datasets'

def create_directory_if_does_not_exist(directory: str) -> None:
    if os.path.exists(directory):
        return

    os.mkdir(directory)


current_directory = os.path.dirname(os.path.realpath(__file__))
root_directory = os.path.dirname(current_directory)
dataset_directory = os.path.join(root_directory, DATASETS_FOLDER_NAME)


train_dataset = TorchDataset(root=dataset_directory, image_set='train', download=True)
