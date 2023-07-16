from dataclasses import dataclass
from enum import Enum

from typing import Callable, Sequence

from torch import nn, Tensor

@dataclass(frozen=True)
class DatasetID:
    name: str
    sub_name: str = None

class DatasetNames(Enum):
    WIKI_TEXT2 = DatasetID(name='wikitext', sub_name='wikitext-2-v1')
    WIKI_TEXT103 = DatasetID(name='wikitext', sub_name='wikitext-103-v1')
    TINY_STORIES = DatasetID(name='skeskinen/TinyStories-hf')
    NUMBERS = DatasetID(name='numbers')

    def __repr__(self):
        if self.value.sub_name:
            return self.value.name + '/' + self.value.sub_name

        return self.value.name

@dataclass(frozen=True)
class Config:
    # Dataset configuration
    dataset: DatasetNames
    epochs: int
    lr: float
    betas: [float]
    batch_size: int
    gradient_clip_value: float
    num_workers: int

    # Model configuration
    accelerator: str
    d_embed: int
    d_proj: int
    d_params: int
    n_hidden: int
    n_layers: int
    n_heads: int
    p_dropout: float
    max_seq_len: int
    ignore_index: int
    activation: Callable[[Tensor], Tensor]
    use_embedding_dropout: bool
    shuffle: bool

    # Logging
    should_log: bool
    log_interval: int

config = Config(
    dataset = DatasetNames.TINY_STORIES,
    epochs = 100,
    lr = 3e-4,
    betas = (0.9, 0.95),
    batch_size = 20,
    gradient_clip_value=0.5,
    num_workers = 8,
    accelerator='gpu',
    d_embed = 512,
    d_proj = 1_024,
    d_params = 1,
    n_hidden = 0,
    n_layers = 12,
    n_heads = 8,
    p_dropout = 0.2,
    max_seq_len = 512,
    ignore_index = 0,
    activation = nn.GELU,
    use_embedding_dropout = False,
    shuffle = True,
    should_log=True,
    log_interval=2_000
)


def get_config():
    return config
