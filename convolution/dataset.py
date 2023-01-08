import os

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

num_workers=8
batch_size=8

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.dirname(BASE_PATH)
DATASET_PATH = os.path.join(ROOT_PATH, 'datasets')

train_dataset = MNIST(root=DATASET_PATH, train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root=DATASET_PATH, train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
