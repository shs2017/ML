"""Training and model related configuration"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Logging
    log: bool
    log_interval: int

    # Training related configuration
    train_epochs: int
    test_epochs: int
    debug: bool

    lr: float
    gamma: float
    epsilon_start: float
    epsilon_min: float

    @property
    def epsilon_step_size(self):
        epsilon_delta = self.epsilon_start - self.epsilon_min
        return epsilon_delta / (self.train_epochs / 2)

config = Config(
    train_epochs = 5_000_000,
    test_epochs = 10,
    debug = False,
    gamma = 0.95,
    lr = 0.001,
    epsilon_start = 1,
    epsilon_min = 0.1,
    log = True,
    log_interval = 25_000
)

def get_config():
    return config
