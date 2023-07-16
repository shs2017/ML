""" Components for transformer based next token prediction

    Key:
        * N: batch size
        * S: sequence length
        * T: target sequence length
        * E: embedding dimension size
        * H: number of heads
        * P: projected embeding dimension size
"""

from typing import Any, Callable, Optional, Sequence, Tuple

from math import exp, log

import torch
from torch import nn, Tensor

from config import get_config
from model.blocks import Decoder
from model.embeddings import Embedding, PositionalEncoding

import pytorch_lightning as pl

config = get_config()

class LightningAttentionModel(pl.LightningModule):
    """PyTorch Lightning wrapper for the Transformer-based decoder prediction model"""

    def __init__(
        self,
        vocab_size: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.lr = config.lr
        self.vocab_size = vocab_size

        self.model = AttentionModel(vocab_size=vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def predict(self, batch: Tuple[Tensor, Tensor], reset: bool) -> Tensor:
        data, targets = batch

        output = self.model(data, use_mask=True)
        loss = self.criterion(output.flatten(end_dim=1), targets.flatten())

        perplexity = exp(loss)
        self.print_loss("perplexity", perplexity)

        return loss

    def print_loss(self, key: str, value: Any) -> None:
        self.log(key, value, batch_size=config.batch_size, prog_bar=True)

    # Standard LightningModule methods
    def training_step(self, batch, _):
        optimizer = self.optimizers()

        loss = self.predict(batch, True)
        self.manual_backward(loss)

        self.clip_gradients(
            optimizer,
            gradient_clip_val=config.gradient_clip_value,
            gradient_clip_algorithm='norm'
        )

        optimizer.step()
        optimizer.zero_grad()

        return loss

    def validation_step(self, batch, _):
        loss = self.predict(batch, False)
        self.print_loss("val_loss", loss)

        perplexity = exp(loss)
        self.print_loss("val_perplexity", perplexity)

        return loss

    def test_step(self, batch, _):
        return self.predict(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


class AttentionModel(nn.Module):
    """Transformer-based decoder prediction model"""

    def __init__(self, vocab_size: int):
        super().__init__()

        self.embedding = Embedding(vocab_size)
        self.embedding_dropout = nn.Dropout1d(p = config.p_dropout) if config.use_embedding_dropout else nn.Identity()
        self.decoder_blocks = nn.ModuleList(
            [
                Decoder()
                for _ in range(config.n_layers)
            ]
        )
        self.output_layer = nn.Linear(config.d_embed, vocab_size)

        self.weights_init_uniform()
        self.generate_next_token_mask(config.max_seq_len)

    def weights_init_uniform(self) -> None:
        init_range = (-0.02, 0.02)
        self.output_layer.weight.data.uniform_(*init_range)
        self.output_layer.bias.data.zero_()

    def generate_next_token_mask(self, seq_len: int) -> None:
        """Mask layer for the self-attention layer

        Example applied mask (query x key = 5 x 5)

        *----Mask Tensor----*
        | F,  T,  T,  T,  T |
        | F,  F,  T,  T,  T |
        | F,  F,  F,  T,  T |
        | F,  F,  F,  F,  T |
        | F,  F,  F,  F,  F |
        *-------------------*

        T are replaced and F are left same in applied matrix
        """
        seq_len -= 1
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.to(bool).cuda()

        self.register_buffer("mask", mask)

    def forward(self, x: Tensor, use_mask: bool) -> Tensor:
        """
        Args:
            x (Tensor): input tensor of shape (N, L, E)
            use_mask (bool): use next token prediction mask if true,
                             otherwise do not use a mask

        Returns:
            A tensor of shape (N, L, E)
        """
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, self.mask)

        return self.output_layer(x)
