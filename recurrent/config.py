from dataclasses import dataclass


@dataclass
class Config:
    d_in: int
    d_hid: int
    d_out: int
    n_layers: int
    vocab_size: int
    device: str
