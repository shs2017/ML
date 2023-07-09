import numpy as np

from abc import ABC, abstractmethod
from itertools import product
import random

import numpy as np

from config import get_config

import pickle

from collections import defaultdict
from typing import Any

config = get_config()

class Agent(ABC):
    def __init__(self, action_shape, observation_shape, lr: float):
        self.table = defaultdict(lambda : np.zeros(action_shape))
        self.observation_shape = observation_shape

        self.lr = lr
        self.gamma = config.gamma

    @abstractmethod
    def update(self, state: any, action: int, reward: float, next_state: any, epsilon: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def select_action(self, state: Any, epsilon: float) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset_state(self):
        raise NotImplementedError

    def greedy_sample(self, state: Any) -> int:
        rewards = self.table[state]
        return np.argmax(rewards)

    def epsilon_sample(self, state: Any, epsilon: float) -> int:
        p = random.random()
        if p < epsilon:
            rewards = self.table[state]
            return random.randrange(len(rewards))

        return self.greedy_sample(state)

    def get_policy(self) -> np.ndarray:
        policy_shape = None
        policy_shape = [space.n for space in self.observation_shape.spaces]

        policy = np.zeros(policy_shape)

        states = product(*[range(shape) for shape in policy_shape])
        for state in states:
            policy[state] = self.greedy_sample(state)

        return policy

    def load(self, path: str) -> None:
        # pickling a defaultdict with a lambda is not possible
        # so we copy this to a seperate dictionary here
        table_copy = {}
        with open(path, 'rb') as f:
            table_copy = pickle.load(f)

        for key, value in table_copy.items():
            self.table[key] = value

    def save(self, path: str) -> None:
        # pickling a defaultdict with a lambda is not possible
        # so we copy this to a seperate dictionary here
        table_copy = {key: value for key, value in self.table.items()}

        with open(path, 'wb') as f:
            pickle.dump(table_copy, f, pickle.HIGHEST_PROTOCOL)

class QLearner(Agent):
    def __init__(self, action_shape, observation_shape, lr: float):
        super().__init__(action_shape, observation_shape, lr)

    def update(self, state: any, action: int, reward: float, next_state: any, epsilon: float) -> int:
        current_rewards = self.table[state]
        future_rewards = self.table[next_state]

        max_td = np.max(future_rewards - current_rewards[action])
        delta = reward + self.gamma * max_td
        current_rewards[action] += self.lr * delta

    def select_action(self, state: Any, epsilon: float) -> int:
        return self.epsilon_sample(state, epsilon)

    def reset_state(self):
        pass

    def __str__(self):
        return 'Q-Learning'

class SARSA(Agent):
    def __init__(self, action_shape, observation_shape, lr: float):
        super().__init__(action_shape, observation_shape, lr)
        self.next_action = None

    def update(self, state: any, action: int, reward: float, next_state: any, epsilon: float) -> int:
        current_rewards = self.table[state]
        future_rewards = self.table[next_state]

        self.next_action = self.epsilon_sample(next_state, epsilon)

        td = future_rewards[self.next_action] - current_rewards[action]
        delta = reward + self.gamma * td
        current_rewards[action] += self.lr * delta

    def select_action(self, state: Any, epsilon: float) -> int:
        if self.next_action:
            return self.next_action

        return self.epsilon_sample(state, epsilon)

    def reset_state(self):
        self.next_action = None

    def __str__(self):
        return 'SARSA'
