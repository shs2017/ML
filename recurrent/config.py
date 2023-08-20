from dataclasses import dataclass


@dataclass
class Config:
    d_hid: int
    d_state: int
    d_out: int
    n_layers: int
