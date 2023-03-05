"""Training and model related configuration"""

from dataclasses import dataclass

@dataclass
class ConfigurationData:
    """Model and training configuration"""

    # Training related configuration
    action_space_shape: int
    train_epochs: int
    test_epochs: int
    debug: bool

    lr: float
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_step_size: float

    # Logging
    log: bool
    log_interval: int


config = None
config_built = False

def build_config(action_size):
    global config
    global config_built

    config_built = True
    config = ConfigurationData(action_space_shape = action_size,
                               train_epochs = 100_000,
                               test_epochs = 10,
                               debug = False,
                               gamma = 1,
                               lr = 1e-1,
                               epsilon_start = 1,
                               epsilon_min = 0.1,
                               epsilon_step_size = 0.001,
                               log = True,
                               log_interval = 25_000)

def get_config():
    global config
    global config_built

    if not config_built:
        raise ValueError('You must build the configuration before using it')

    return config
