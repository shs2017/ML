from config import get_config
from math import exp

config = get_config()

class EpsilonScheduler:
    def __init__(self):
        self.epsilon = config.initial_epsilon
        self.steps = 0

    def step(self):
        self.epsilon = config.min_epsilon + (config.initial_epsilon - config.min_epsilon) * exp(-1. * self.steps / config.epsilon_steps)
        self.steps += 1
