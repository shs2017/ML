""" Trains on transformer-based token prediction model """

from typing import Callable
from math import exp

import torch
from torch import nn, Tensor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model import AttentionModel
from dataset import dataset

from config import get_config

config = get_config()

class MainModel(pl.LightningModule):
    """PyTorch Lightning wrapper for the Transformer-based decoder prediction model"""

    def __init__(
        self,
        n_tokens: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = config.lr
        self.n_tokens = n_tokens

        self.model = AttentionModel(n_tokens=n_tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)

    def _print_loss(self, key, value):
        self.log(key, value, batch_size=config.batch_size, prog_bar=True)

    def _predict(self, batch):
        data, targets = batch

        output = self.model(data, use_mask=True)
        loss = self.criterion(output.view(-1, self.n_tokens), targets)

        perplexity = exp(loss)
        self._print_loss("perplexity", perplexity)

        return loss

    # Standard LightningModule methods implemented
    def training_step(self, batch, _):
        return self._predict(batch)

    def validation_step(self, batch, _):
        loss = self._predict(batch)
        self._print_loss("val_loss", loss)

        perplexity = exp(loss)
        self._print_loss("val_perplexity", perplexity)

        return loss

    def test_step(self, batch, _):
        return self._predict(batch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)

if __name__ == '__main__':
    print(f'Config = {config}')

    logger = None
    if config.should_log:
        logger = TensorBoardLogger("logs", name="transformer")

    model = MainModel(n_tokens=dataset.vocab_size)
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=config.epochs,
        log_every_n_steps=config.log_interval,
        gradient_clip_val=config.gradient_clip_value,
        logger=logger,
        callbacks=[],
    )

    trainer.fit(model=model, datamodule=dataset)
    trainer.validate(datamodule=dataset)
