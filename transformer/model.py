""" Components for transformer based next token prediction 

    Key:
        * N: batch size
        * S: sequence length
        * T: target sequence length 
        * E: embedding dimension size
        * H: number of heads
        * P: projected embeding dimension size
"""

from typing import Callable, Sequence

from math import sqrt, log

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """Pre-computed sinusoidal positional embeddings"""

    def __init__(self, d_embed: int, max_seq_len: int, p_dropout: float):

        """
        Args:
            d_embed (int): embedding dimension size
            max_seq_len (int): maximum sequence length
            p_dropout (float): dropout probability


        Formula:
            positional_encoding(pos, 2 * i)     = sin(pos / 10000**((2 * i) / d_embed))
            positional_encoding(pos, 2 * i + 1) = cos(pos / 10000**((2 * i) / d_embed))
        """
        super().__init__()

        numerator = torch.arange(0, max_seq_len)
        denominator = torch.exp(-torch.arange(0, d_embed, 2) * log(10_000) / d_embed)

        phase = numerator.unsqueeze(1) * denominator.unsqueeze(0)

        encoding = torch.zeros((max_seq_len, d_embed))
        encoding[:, 0::2] = torch.sin(phase)
        encoding[:, 1::2] = torch.cos(phase)

        encoding = encoding.unsqueeze(0)  # add batch dimension
        self.register_buffer("positional_encoding", encoding)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor):
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


class Embedding(nn.Module):
    """Token Sequence Embeddings"""

    def __init__(
        self,
        d_embed: int,
        max_seq_len: int,
        n_tokens: int,
        p_dropout: float,
        unk_index: int,
    ):
        """
        Args:
            d_embed (int): embedding dimension
            max_seq_len (int): maximum sequence length
            n_tokens (int): number of tokens
            p_dropout (float): dropout probrability
            unk_index (int): vocab index to use for `<unk>` token
        """
        super().__init__()
        self.embed = nn.Embedding(n_tokens, d_embed, padding_idx=unk_index)
        self.positional_encoding = PositionalEncoding(
            d_embed, max_seq_len=max_seq_len, p_dropout=p_dropout
        )
        self.sqrt_d_embed = sqrt(d_embed)
        self._weights_init_uniform()

    def _weights_init_uniform(self):
        init_range = (-0.1, 0.1)
        self.embed.weight.data.uniform_(*init_range)

    def forward(self, indices: Tensor):
        """
        Args:
            indices (Tensor): LongTensor of shape (N, S) where each element is the index of an embedding

        Returns:
            A concatenated tensor of shape (N, S, E) with the indices of the input replaced with embeddings.
        """
        src = self.embed(indices) * self.sqrt_d_embed
        return self.positional_encoding(src)


class MLP(nn.Module):
    """Multi-layer perceptron with dropout"""

    def __init__(
        self,
        d_embed: int,
        d_proj: int,
        n_hidden: int,
        activation: Callable[[Tensor], Tensor],
        p_dropout: float,
    ):
        """
        Args:
            d_embed (int): input and output layer dimensions
            d_proj (int): mlp hidden layer dimensions
            n_hidden (int): number of hidden layers
            activation (Callable[[Tensor], Tensor]): activation function placed after each weight multiplication
            p_dropout (float): dropout probrability
        """
        super().__init__()

        layers = []

        layers.append(nn.Linear(d_embed, d_proj))
        layers.append(activation())
        layers.append(nn.Dropout(p_dropout))

        for _ in range(n_hidden):
            layers.append(nn.Linear(d_proj, d_proj))
            layers.append(activation())
            layers.append(nn.Dropout(p_dropout))

        layers.append(nn.Linear(d_proj, d_embed))
        layers.append(nn.Dropout(p_dropout))

        self.f = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): tensor of shape (..., E)

        Returns:
            A tensor outputted by the mlp of the same shape except the last dimension which
            will be `features[-1]` in size
        """
        return self.f(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with pre-layer norm"""

    def __init__(self, d_embed: int, n_heads: int, p_dropout: float):
        """
        Args:
            d_embed (int): embedding dimension
            n_heads (int): number of attention heads
            p_dropout (float): dropout probrability
        """
        super().__init__()

        assert d_embed % n_heads == 0

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.p_embed = d_embed // n_heads

        self.sqrt_d_embed = sqrt(d_embed)

        # Projection layers for key, query, and values
        self.P_k = nn.Linear(d_embed, d_embed)
        self.P_q = nn.Linear(d_embed, d_embed)
        self.P_v = nn.Linear(d_embed, d_embed)

        self.proj = nn.Linear(d_embed, d_embed)

        self.att_dropout = nn.Dropout(p_dropout)
        self.logits_dropout = nn.Dropout(p_dropout)

    def forward(self, key: Tensor, query: Tensor, value: Tensor, mask: Tensor):
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
        Q = self.P_q(key).view(N, T, self.n_heads, self.p_embed)
        V = self.P_v(key).view(N, S, self.n_heads, self.p_embed)

        # Permute matrix so that the matrix multiplication dimensions work out
        K = K.transpose(1, 2)  # (N, S, H, P) -> (N, H, S, P)
        Q = Q.transpose(1, 2)  # (N, T, H, P) -> (N, H, T, P)
        V = V.transpose(1, 2)  # (N, S, H, P) -> (N, H, S, P)

        # Compute attention matrix
        att = Q @ K.transpose(-1, -2) / self.sqrt_d_embed  # (N, H, T, S)
        if mask != None:
            att.masked_fill_(mask, float("-inf"))

        # Multiply with values and reshape
        out = torch.softmax(att, dim=-1) @ V  # (N, H, T, P)

        # Dropout
        out = self.att_dropout(out)

        out = out.transpose(1, 2).contiguous().view(N, T, E)  # (N, T, E)
        return self.logits_dropout(self.proj(out))


class SelfAttention(nn.Module):
    """Self-attention module"""

    def __init__(self, d_embed: int, n_heads: int, p_dropout: float):
        """
        Args:
            d_embed (int): embedding dimension
            n_heads (int): number of attention heads
            p_dropout (float): dropout probrability
        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_embed, n_heads, p_dropout)

    def forward(self, x: Tensor, mask: Tensor):
        """
        Args:
            x (Tensor): input tensor of shape (N, S, E)
            mask (Tensor): boolean mask tensor of shape (T, S) where
                           True means to replace the element and False
                           means to leave it as is

        Returns:
            A tensor outputted by multi-headed attention of shape (N, T, E)
        """
        return self.multi_head_attention(x, x, x, mask)


class DecoderBlock(nn.Module):
    """A transformer-based decoder block"""

    def __init__(
        self,
        d_embed: int,
        d_proj: int,
        n_hidden: int,
        n_heads: int,
        p_dropout: float,
        activation: Callable[[Tensor], Tensor] = nn.ReLU,
    ):
        """
        Args:
            d_embed (int): embedding dimension
            d_proj (int): projected dimension of the mlp
            n_hidden (int): number of hidden layers to use in mlp
            n_heads (int): number of attention heads
            p_dropout (float): dropout probrability
            activation (Callable[[Tensor], Tensor]): activation function placed after each weight multiplication
        """
        super().__init__()

        self.att_layer_norm = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(d_embed, n_heads, p_dropout)

        self.mlp_layer_norm = nn.LayerNorm(d_embed)
        self.mlp = MLP(d_embed, d_proj, n_hidden, activation, p_dropout)

    def forward(self, x: Tensor, mask: Tensor):
        """
        Args:
            x (Tensor): input tensor of shape (N, S, E)
            mask (Tensor): boolean mask tensor of shape (T, S) where
                           True means to replace the element and False
                           means to leave it as is

        Returns:
            A tensor outputted by decoder block of shape (N, T, E)
        """
        x = self.att_layer_norm(x)
        x = x + self.attention(x, mask=mask)

        x = self.mlp_layer_norm(x)
        x = x + self.mlp(x)

        return x


class AttentionModel(nn.Module):
    """Transformer-based decoder prediction model"""

    def __init__(
        self,
        n_layers: int,
        d_embed: int,
        d_proj: int,
        n_hidden: int,
        n_heads: int,
        p_dropout: float,
        max_seq_len: int,
        n_tokens: int,
        unk_index: int = 0,
        activation: Callable[[Tensor], Tensor] = nn.ReLU,
    ):
        """
        Args:
            n_layers (int): number of decoder blocks to use
            d_embed (int): embedding dimension
            d_proj (int): mlp hidden layer dimensions
            n_hidden (int): number of hidden layers of mlp
            n_heads (int): number of attention heads
            p_dropout (float): dropout probrability
            max_seq_len (int): maximum sequence length
            n_tokens (int): number of tokens
            unk_index (int): vocab index to use for `<unk>` token
            activation (Callable[[Tensor], Tensor]): activation function placed after each weight multiplication
        """
        super().__init__()

        self.embedding = Embedding(d_embed, max_seq_len, n_tokens, p_dropout, unk_index)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(d_embed, d_proj, n_hidden, n_heads, p_dropout, activation)
                for _ in range(n_layers)
            ]
        )
        self.output_layer = nn.Linear(d_embed, n_tokens)

        self._weights_init_uniform()
        self._generate_next_token_mask(max_seq_len)

    def _weights_init_uniform(self):
        init_range = (-0.1, 0.1)
        self.output_layer.weight.data.uniform_(*init_range)
        self.output_layer.bias.data.zero_()

    def _generate_next_token_mask(self, seq_len: int):
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
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(bool).cuda()
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor, use_mask: bool):
        """
        Args:
            x (Tensor): input tensor of shape (N, L, E)
            use_mask (bool): use next token prediction mask if true,
                             otherwise do not use a mask

        Returns:
            A tensor of shape (N, L, E)
        """

        if use_mask:
            seq_len = x.size(1)
            mask = self.mask[:seq_len, :seq_len]

        x = self.embedding(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, mask)

        return self.output_layer(x)
