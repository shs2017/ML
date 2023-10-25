import pytorch_lightning as pl

import torch.nn.functional as F

from torch import optim

from math import exp

from config import Config, get_config
from dataset import SegmentationDataset
from model import UNet

class MainModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()

        self.lr = config.lr
        self.momentum = config.momentum

        self.model = UNet(config)

    def compute_loss(self, batch):
        image, true_segmentation = batch
        true_segmentation = true_segmentation.squeeze(1)

        predicted_segmentation = self.model(image)
        loss = F.cross_entropy(predicted_segmentation, true_segmentation)
        return loss

    def training_step(self, batch, _):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ppl', exp(loss), prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ppl', exp(loss), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )

config = get_config()

main_model = MainModel(config)

dataset = SegmentationDataset(config)
train_dataset = dataset.retrieve_train_data()
val_dataset = dataset.retrieve_val_data()

trainer = pl.Trainer(max_epochs=config.max_epochs, log_every_n_steps=config.log_every_n_steps)
trainer.fit(model=main_model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)
