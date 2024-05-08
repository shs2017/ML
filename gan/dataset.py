from config import Config

from torch.utils.data import DataLoader

from torchvision.datasets import Food101
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


transform = Compose([
    Resize(size=(64, 64)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_dataloader(config: Config, train: bool):
    split = 'train' if train else 'test'
    train_dataset = Food101(root='../datasets', split=split, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    return train_dataloader