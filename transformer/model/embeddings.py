from math import log, sqrt

import torch
from torch import nn, Tensor

from config import get_config

config = get_config()

class Embedding(nn.Module):
    """Token Sequence Embeddings"""

    def __init__(self, vocab_size: int):
        super().__init__()

        self.embed = nn.Embedding(vocab_size,
                                  config.d_embed,
                                  padding_idx=config.ignore_index)

        self.positional_encoding = PositionalEncoding()
        self.sqrt_d_embed = sqrt(config.d_embed)
        self._weights_init_uniform()

    def _weights_init_uniform(self) -> None:
        init_range = (-0.1, 0.1)
        self.embed.weight.data.uniform_(*init_range)

    def forward(self, indices: Tensor) -> Tensor:
        """
        Args:
            indices (Tensor): LongTensor of shape (N, S) where each element is the index of an embedding

        Returns:
            A concatenated tensor of shape (N, S, E) with the indices of the input replaced with embeddings.
        """
        src = self.embed(indices) * self.sqrt_d_embed
        return self.positional_encoding(src)

class PositionalEncoding(nn.Module):
    """Pre-computed sinusoidal positional embeddings"""

    def __init__(self):

        """
        Formula:
            positional_encoding(pos, 2 * i)     = sin(pos / 10000**((2 * i) / d_embed))
            positional_encoding(pos, 2 * i + 1) = cos(pos / 10000**((2 * i) / d_embed))
        """
        super().__init__()

        max_len = 5_000
        # max_len = config.max_seq_len

        numerator = torch.arange(0, max_len)
        denominator = torch.exp(-torch.arange(0, config.d_embed, 2) * log(10_000) / config.d_embed)

        phase = numerator.unsqueeze(1) * denominator.unsqueeze(0)

        encoding = torch.zeros((max_len, config.d_embed))
        encoding[:, 0::2] = torch.sin(phase)
        encoding[:, 1::2] = torch.cos(phase)

        encoding = encoding.unsqueeze(0)  # add batch dimension
        self.register_buffer("positional_encoding", encoding)
        self.dropout = nn.Dropout(config.p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor of shape (N, S, E),
                        where S is smaller than or equal to
                        the maximum sequence length

        Returns:
            A tensor of the same shape as the input with the
            positional encoding added it
        """
        seq_len = x.size(1)
        x = self.positional_encoding[:, :seq_len] + x

        return self.dropout(x)
