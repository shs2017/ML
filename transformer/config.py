from dataclasses import dataclass
from enum import Enum

from typing import Callable, Sequence

from torch import nn, Tensor

class DatasetNames(Enum):
    WIKI_TEXT2 = 'WikiText2'
    WIKI_TEXT103 = 'WikiText103'

    def __repr__(self):
        return self.value

@dataclass(frozen=True)
class Config:
    # Dataset configuration
    dataset: DatasetNames
    epochs: int
    lr: float
    batch_size: int
    gradient_clip_value: float

    # Modal configuration
    d_embed: int
    d_proj: int
    n_hidden: int
    n_layers: int
    n_heads: int
    p_dropout: float
    max_seq_len: int
    ignore_index: int
    activation: Callable[[Tensor], Tensor]

    # Logging
    should_log: bool
    log_interval: int

config = Config(
    dataset = DatasetNames.WIKI_TEXT103,
    epochs = 100,
    lr = 1e-3,
    batch_size = 20,
    gradient_clip_value=0.5,
    d_embed = 400,
    d_proj = 1_000,
    n_hidden = 0,
    n_layers = 16,
    n_heads = 10,
    p_dropout = 0.2,
    max_seq_len = 256,
    ignore_index = 0,
    activation = nn.ReLU,
    should_log=True,
    log_interval=2_000
)

def get_config():
    return config
