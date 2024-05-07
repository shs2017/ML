import pytorch_lightning as pl

from config import get_config
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

from model import MainModel

if __name__ == '__main__':
    config = get_config()

    transform = Compose([Resize(size=(64, 64)), ToTensor()])
    train_dataset = MNIST(root='../datasets', train=True, transform=transform, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    main_model = MainModel(config)

    trainer = pl.Trainer(max_epochs=config.max_epochs, log_every_n_steps=config.log_every_n_steps)
    trainer.fit(model=main_model, train_dataloaders=train_dataloader)

    main_model.cuda().generate_image()