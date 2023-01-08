import pytorch_lightning as pl

import torch.nn.functional as F

import torch
from torch import optim

from model import Residuals
from dataset import train_dataloader

class MainModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Residuals()
        self.lr = 1e-5
        self.momentum = 0.9

    def training_step(self, batch, _):
        img, labels = batch

        prediction = self.model(img)
        loss = F.cross_entropy(prediction, labels)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(),
                         lr=self.lr,
                         momentum=self.momentum)

main_model = MainModel()

trainer = pl.Trainer(max_epochs=25, log_every_n_steps=100)
trainer.fit(model=main_model, train_dataloaders=train_dataloader)
