from abc import ABC, abstractmethod
from collections import defaultdict
from random import randrange, random
from typing import List, Tuple

from config import get_config
from context import Context
from utils import argmax

import numpy as np

config = get_config()

class AbstractStrategy(ABC):
    """Abstract class for the learning algorithm"""

    @abstractmethod
    def action(self, context: Context, epsilon: float) -> int:
        """Generate the next action based on the current context"""
        raise NotImplementedError

    @abstractmethod
    def update(self, context: Context, epsilon: float):
        """Update the current model based on the given context"""
        raise NotImplementedError

    @property
    def policy_maps(self) -> Tuple[List[List[int]]]:
        """Graph the policy on a heatmap"""
        # TODO: Make this not hard-coded
        usable_ace_policy = np.empty((11, 22))
        no_usable_ace_policy = np.empty((11, 22))
        usable_ace_policy.fill(-1)
        no_usable_ace_policy.fill(-1)

        for state, action in self.q.q.items():
            player_hand = state[0]
            dealer_hand = state[1]
            usable_ace = state[2]
            best_action = argmax(action)

            # is this an indicator of the failure cases
            if player_hand > 21 or dealer_hand > 11:
                continue

            if usable_ace:
                usable_ace_policy[dealer_hand, player_hand] = best_action
            else:
                no_usable_ace_policy[dealer_hand, player_hand] = best_action

        return usable_ace_policy, no_usable_ace_policy

class Agent:
    """Class representing an agent"""

    def __init__(self, learner: AbstractStrategy):
        self.learner = learner
        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_step_size = config.epsilon_step_size

    def train_mode(self) -> None:
        self.train_mode = True
        self.epsilon = self.epsilon_start

    def test_mode(self) -> None:
        self.train_mode = False
        self.epsilon = 0

    def action(self, context: Context) -> int:
        return self.learner.action(context, self.epsilon)

    def update(self, context: Context) -> None:
        self.learner.update(context, self.epsilon)

    def end_epoch(self) -> None:
        if self.train_mode:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step_size)

    @property
    def policy_maps(self) -> Tuple[List[List[int]]]:
        return self.learner.policy_maps

class SARSA(AbstractStrategy):
    """SARSA implementation"""

    def __init__(self, action_space_shape: int):
        self.q = QFunction(action_space_shape)

    def action(self, context: Context, epsilon: float) -> int:
        return self.q.greedy_epsilon(context, epsilon=epsilon)

    def update(self, context: Context, epsilon: float) -> None:
        context.next_action = self.q.greedy_epsilon(context.next_state, epsilon=epsilon)
        self.q.bellman_update(context, config.lr, config.gamma)


class QLearner(AbstractStrategy):
    """Q-learning implementation"""

    def __init__(self, action_space_shape: int):
        self.q = QFunction(action_space_shape)

    def action(self, context: Context, epsilon) -> int:
        return self.q.greedy_epsilon(context, epsilon=epsilon)

    def update(self, context: Context, epsilon=None) -> None:
        context.next_action = self.q.greedy_epsilon(context.next_state, epsilon=None)
        self.q.bellman_update(context, config.lr, config.gamma)

class QFunction:
    def __init__(self, action_space_shape: int):
        self.action_space_shape = action_space_shape
        self.q = defaultdict(lambda : [0.] * self.action_space_shape)

    def greedy_epsilon(self, state, epsilon=None) -> int:

        r = random()
        if epsilon != None and r < epsilon:
            return randrange(0, self.action_space_shape)
        else:
            return argmax(self.q[state])

    def bellman_update(self, context, lr, gamma) -> None:
        q_current = self.get(context.state, context.action)
        q_future = 0.
        if not context.terminated:
            q_future = self.get(context.next_state, context.next_action)
        td_error = context.reward + gamma * q_future - q_current

        new_value = q_current + lr * td_error

        self.update(context.state, context.action, new_value)

    def get(self, state, action_index) -> float:
        """Override for getting the q value for a
           given state and an action's index """
        actions = self.q[state]
        out = actions[action_index]
        return out

    def update(self, state, action_index, new_value) -> None:
        """Override for getting the q value for a
           given state and an action's index """
        actions = self.q[state]
        actions[action_index] = new_value
