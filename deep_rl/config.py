from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Environment
    env_name: str
    n_episodes: int
    seed: int
    log_interval: int
    frames_per_state: int

    # Model
    d_hid: int

    # Training
    batch_size: int
    gamma: float
    initial_epsilon: float
    epsilon_steps: float
    min_epsilon: float
    replay_buffer_size: int
    lr: float
    betas: [float]
    tau: float
    clip_value: float

config = Config(
    env_name = 'PongNoFrameskip-v4',
    n_episodes = 2_000,
    seed = 42,
    log_interval = 100,
    frames_per_state = 4,
    d_hid = 128,
    batch_size = 128,
    gamma = 0.99,
    initial_epsilon = 0.9,
    epsilon_steps = 100_000,
    min_epsilon = 0.05,
    replay_buffer_size = 10_000,
    lr = 1e-4,
    betas = (0.9, 0.999),
    tau = 0.995,
    clip_value = 100
)

def get_config():
    return config
