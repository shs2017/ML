""" Trains on transformer-based token prediction model """

from typing import Callable
from math import exp

import torch

from torch import nn, Tensor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from model import AttentionModel
from dataset import dataset, BATCH_SIZE


class EMALossCalculator:
    """Exponential Moving Average Loss Tracker"""

    def __init__(self, alpha: float):
        """
        Args:
            alpha (float): represents how much the new loss should contribute to average
        """
        if 1 < alpha <= 0:
            raise ValueError("alpha must be in interval (0, 1]")

        self.loss = 0.0
        self.alpha = alpha

    def update(self, loss: float):
        """Updates moving average and returns the new loss

        Args:
            loss (float): loss value to apply EMA against internally stored loss

        Returns:
            float: containing updated EMA loss
        """
        self.loss = self.alpha * loss + (1 - self.alpha) * loss
        return self.loss


class MainModel(pl.LightningModule):
    """PyTorch Lightning wrapper for the Transformer-based decoder prediction model"""

    def __init__(
        self,
        lr: float,
        n_tokens: int,
        d_embed: int,
        d_proj: int,
        n_hidden: int,
        n_layers: int,
        n_heads: int,
        p_dropout: float,
        max_seq_len: int,
        activation: Callable[[Tensor], Tensor] = nn.ReLU,
        ignore_index: int = 0,
        loss_calculator=None,
    ):
        """
        Args:
            lr (float): learning rate
            n_tokens (int): number of tokens
            d_embed (int): embedding dimension
            d_proj (int):  mlp hidden dimension
            n_hidden (int): number of mlp hidden layers
            n_layers (int): number of decoder blocks to use
            n_heads (int): number of attention heads
            p_dropout (float): dropout probrability
            max_seq_len (int): maximum sequence length
            activation (Callable[[Tensor], Tensor]): mlp activation function (defaults to relu)
            ignore_index (int): vocab index used for the "<unk>" token
            loss_calculator: class used for updating the loss. Leave as `None` if
                             the computed loss does not need to be changed.
        """
        super().__init__()

        self.loss_calculator = loss_calculator
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.n_tokens = n_tokens

        self.model = AttentionModel(
            n_layers=n_layers,
            d_embed=d_embed,
            d_proj=d_proj,
            n_hidden=n_hidden,
            n_heads=n_heads,
            p_dropout=p_dropout,
            max_seq_len=max_seq_len,
            n_tokens=n_tokens,
            unk_index=ignore_index,
            activation=activation,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _print_loss(self, key, value):
        self.log(key, value, batch_size=BATCH_SIZE, prog_bar=True)

    def _predict(self, batch):
        """Predict next tokens for each batch"""
        data, targets = batch

        output = self.model(data, use_mask=True)
        loss = self.criterion(output.view(-1, self.n_tokens), targets)

        if self.loss_calculator is not None:
            loss = self.loss_calculator.update(loss)

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


vocab_size = dataset.vocab_size
ignore_index = dataset.ignore_index

callbacks = []

model = MainModel(
    lr=1.0,
    n_tokens=vocab_size,
    d_embed=256,
    d_proj=512,
    n_hidden=2,
    n_layers=4,
    n_heads=8,
    p_dropout=0.2,
    max_seq_len=35,
    ignore_index=ignore_index,
    loss_calculator=None,  # put EMALossCalculator(alpha=...) here to use EMA loss instead
)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=5,
    log_every_n_steps=200,
    gradient_clip_val=0.5,
    callbacks=callbacks,
)

trainer.fit(model=model, datamodule=dataset)
trainer.validate(datamodule=dataset)
