from time import time

from torch import optim
from torch import nn

import torch.nn.functional as F
import pytorch_lightning as pl

from config import Config
from model import SimpleMLP
from dataset import xor_dataloader

class MainModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config.lr
        self.model = SimpleMLP(config)

    def training_step(self, batch, batch_idx):
        input_data, ground_truth = batch
        prediction = self.model(input_data)

        loss = F.binary_cross_entropy_with_logits(prediction, ground_truth)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(),
                         lr=self.lr)


config = Config(lr = 1,
                max_epochs = 300,
                log_every_n_steps = 1,
                d_in =  2,
                d_hid = 4,
                d_out = 1,
                n_hid = 1)

main_model = MainModel(config)
trainer = pl.Trainer(max_epochs=config.max_epochs, log_every_n_steps=config.log_every_n_steps)
trainer.fit(model=main_model, train_dataloaders=xor_dataloader)
