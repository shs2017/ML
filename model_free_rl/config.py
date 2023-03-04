"""Training and model related configuration"""

from dataclasses import dataclass

@dataclass
class Config:
    """Model and training configuration"""

    # Training related configuration
    action_space_shape: int
    epochs: int
    debug: bool

    lr: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_step_size: float

    env: any

    # Agent related configuration
