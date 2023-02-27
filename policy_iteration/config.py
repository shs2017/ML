"""Training and model related configuration"""

from dataclasses import dataclass

@dataclass
class Config:
    action_space_shape: int
    epochs: int
    debug: bool = False
