from dataclasses import dataclass

from torch import nn

@dataclass
class Config():
    # Training Parameters
    lr: float
    max_epochs: int
    log_every_n_steps: int

    # Model Parameters
    d_in: int
    d_hid: int
    d_out: int

    n_hid: int

    activation = nn.Sigmoid
