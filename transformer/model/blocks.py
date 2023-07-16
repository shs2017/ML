import logging
from typing import Optional
from math import log, sqrt

import torch

from torch import nn, Tensor
from config import get_config

config = get_config()

class Decoder(nn.Module):
    """A transformer-based decoder block"""

    def __init__(self):
        super().__init__()

        self.att_layer_norm = nn.LayerNorm(config.d_embed)
        self.attention = SelfAttention(use_softmax=True)

        self.mlp_layer_norm = nn.LayerNorm(config.d_embed)
        self.mlp = MLP()

    def forward(self, context: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            context (Tensor): input tensor of shape (N, S, E)
            mask (Tensor): boolean mask tensor of shape (T, S) where
                           True means to replace the element and False
                           means to leave it as is

        Returns:
            A tensor outputted by decoder block of shape (N, T, E)
        """
        context = context + self.attention(self.att_layer_norm(context), mask=mask)
        context = context + self.mlp(self.mlp_layer_norm(context))

        return context

class SelfAttention(nn.Module):
    """Self-attention module"""

    def __init__(self, use_softmax: bool):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(use_softmax)

    def forward(self, context: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            context (Tensor): input tensor of shape (N, S, E)
            mask (Tensor): boolean mask tensor of shape (T, S) where
                           True means to replace the element and False
                           means to leave it as is

        Returns:
            A tensor outputted by multi-headed attention of shape (N, T, E)
        """
        return self.multi_head_attention(context, context, context, mask=mask)

class MultiHeadAttention(nn.Module):
    """Multi-head attention with pre-layer norm"""

    def __init__(self, use_softmax: bool):
        super().__init__()

        assert config.d_embed % config.n_heads == 0

        self.d_embed = config.d_embed
        self.n_heads = config.n_heads
        self.p_embed = config.d_embed // config.n_heads

        self.sqrt_d_embed = sqrt(config.d_embed)

        # Projection layers for key, query, and values
        self.P_k = nn.Linear(config.d_embed, config.d_embed)
        self.P_q = nn.Linear(config.d_embed, config.d_embed)
        self.P_v = nn.Linear(config.d_embed, config.d_embed)

        self.proj = nn.Linear(config.d_embed, config.d_embed)

        self.att_dropout = nn.Dropout(config.p_dropout)
        self.logits_dropout = nn.Dropout(config.p_dropout)

        self.use_softmax = use_softmax

    def forward(self, key: Tensor, query: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            key (Tensor): key tensor of shape (N, S, E)
            query (Tensor): query tensor of shape (N, T, E)
            value (Tensor): value tensor of shape (N, S, E)
            mask (Tensor): boolean mask tensor of shape (T, S) where
                           True means to replace the element and False
                           means to leave it as is

        Returns:
            A tensor outputted by multi-headed attention of shape (N, T, E)
        """
        N, S, E = key.shape
        _, T, _ = query.shape

        # Split embeddings into multiple heads
        K = self.P_k(key).view(N, S, self.n_heads, self.p_embed)
        Q = self.P_q(query).view(N, T, self.n_heads, self.p_embed)
        V = self.P_v(value).view(N, S, self.n_heads, self.p_embed)

        # Permute matrix so that the matrix multiplication dimensions work out
        K = K.transpose(1, 2)  # (N, S, H, P) -> (N, H, S, P)
        Q = Q.transpose(1, 2)  # (N, T, H, P) -> (N, H, T, P)
        V = V.transpose(1, 2)  # (N, S, H, P) -> (N, H, S, P)

        # Compute attention matrix
        att = Q @ K.transpose(-1, -2) / self.sqrt_d_embed  # (N, H, T, S)
        if mask != None:
            # print(torch.any(att[:, :, -100:] > 1), torch.any(att[:, :, :-100] > 1), torch.max(att))
            att.masked_fill_(mask, float("-inf"))

        # Multiply with values and reshape
        if self.use_softmax:
            att = torch.softmax(att, dim=-1)
        out = att @ V  # (N, H, T, P)

        # Dropout
        out = self.att_dropout(out)

        out = out.transpose(1, 2).contiguous().view(N, T, E)  # (N, T, E)
        return self.logits_dropout(self.proj(out))

class MLP(nn.Module):
    """Multi-layer perceptron with dropout"""

    def __init__(self):
        super().__init__()

        layers = []

        layers.append(nn.Linear(config.d_embed, config.d_proj))
        layers.append(config.activation())
        layers.append(nn.Dropout(config.p_dropout))

        for _ in range(config.n_hidden):
            layers.append(nn.Linear(config.d_proj, config.d_proj))
            layers.append(config.activation())
            layers.append(nn.Dropout(config.p_dropout))

        layers.append(nn.Linear(config.d_proj, config.d_embed))
        layers.append(nn.Dropout(config.p_dropout))

        self.f = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): tensor of shape (..., E)

        Returns:
            A tensor outputted by the mlp of the same shape except the last dimension which
            will be `features[-1]` in size
        """
        return self.f(x)
