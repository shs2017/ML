from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    gamma: float
    initial_epsilon: float
    minimum_epsilon: float
    log_interval: int
    seed: int

config = Config(
    gamma = 0.99,
    initial_epsilon = 0.9,
    minimum_epsilon = 5e-2,
    log_interval = 10_000,
    seed = 42
)

def get_config():
    return config
