""" Trains on transformer-based token prediction model """

from math import exp

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging

from model import AttentionModel
from dataset import dataset


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
        n_layers: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        ignore_index: int = 0,
        loss_calculator=None,
    ):
        """
        Args:
            n_tokens (int): number of tokens
            d_embed (int): embedding dimension
            n_layers (int): number of decoder blocks to use
            n_heads (int): number of attention heads
            dropout (float): dropout probrability
            max_seq_len (int): maximum sequence length
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
            n_heads=n_heads,
            features=[d_embed, d_embed, d_embed],
            max_seq_len=max_seq_len,
            n_tokens=n_tokens,
            dropout=dropout,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _predict(self, batch):
        """Predict next tokens for each batch"""
        data, targets = batch

        output = self.model(data, use_mask=True)
        loss = self.criterion(output.view(-1, self.n_tokens), targets)

        if self.loss_calculator is not None:
            loss = self.loss_calculator.update(loss)

        perplexity = exp(loss)
        self.log("perplexity", perplexity, prog_bar=True)

        return loss

    # Standard PyTorch Lightning LightningModule methods implemented

    def training_step(self, batch, _):
        return self._predict(batch)

    def validation_step(self, batch, _):
        loss = self._predict(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, _):
        return self._predict(batch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)


vocab_size = dataset.vocab_size
ignore_index = dataset.ignore_index

callbacks = [
    StochasticWeightAveraging(5.0),
    EarlyStopping(monitor="val_loss", mode="min"),
]
model = MainModel(
    lr=1.0,
    n_tokens=vocab_size,
    d_embed=200,
    n_layers=5,
    n_heads=8,
    dropout=0.2,
    max_seq_len=35,
    ignore_index=ignore_index,
    loss_calculator=EMALossCalculator(alpha=0.05),
)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=20,
    log_every_n_steps=200,
    gradient_clip_val=0.5,
    callbacks=callbacks,
)

trainer.fit(model=model, datamodule=dataset)
trainer.validate(datamodule=dataset)
