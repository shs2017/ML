from collections import defaultdict
from random import randrange, random
from abc import ABC, abstractmethod

from config import get_config
from context import Context
from utils import argmax

class AbstractStrategy(ABC):
    """Base class for handling the update algorithm
       using the strategy design pattern"""

    @abstractmethod
    def action(self, context: Context, epsilon: float):
        """Generate the next action based on the current context"""
        raise NotImplementedError

    @abstractmethod
    def update(self, context: Context, epsilon: float):
        """Update the current model based on the given context"""
        raise NotImplementedError

class Agent:
    """Class representing an agent"""

    def __init__(self, learner: AbstractStrategy):
        self.learner = learner

        config = get_config()
        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_step_size = config.epsilon_step_size

    def train_mode(self):
        self.train_mode = True
        self.epsilon = self.epsilon_start

    def test_mode(self):
        self.train_mode = False
        self.epsilon = 0

    def action(self, context: Context) -> int:
        return self.learner.action(context, self.epsilon)

    def update(self, context: Context):
        self.learner.update(context, self.epsilon)

    def end_epoch(self):
        if self.train_mode:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step_size)

class QFunction:
    def __init__(self):
        self.action_space_shape  = get_config().action_space_shape
        self.q = defaultdict(lambda : [0.] * self.action_space_shape)

    def greedy_epsilon(self, state, epsilon=None):

        r = random()
        if epsilon != None and r < epsilon:
            return randrange(0, self.action_space_shape)
        else:
            return argmax(self.q[state])

    def bellman_update(self, context, lr, gamma):
        q_current = self.get(context.state, context.action)
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

    def update(self, state, action_index, new_value):
        """Override for getting the q value for a
           given state and an action's index """
        actions = self.q[state]
        actions[action_index] = new_value

class SARSA(AbstractStrategy):
    """SARSA implementation"""

    def __init__(self):
        self.config = get_config()
        self.q = QFunction()

    def action(self, context: Context, epsilon: float) -> int:
        return self.q.greedy_epsilon(context, epsilon=epsilon)

    def update(self, context: Context, epsilon: float):
        context.next_action = self.q.greedy_epsilon(context.state, epsilon=epsilon)
        self.q.bellman_update(context, self.config.lr, self.config.gamma)

class QLearner(AbstractStrategy):
    """Q-learning implementation"""

    def __init__(self):
        self.config = get_config()
        self.q = QFunction()

    def action(self, context: Context, epsilon) -> int:
        return self.q.greedy_epsilon(context, epsilon=epsilon)

    def update(self, context: Context, epsilon=None):
        context.next_action = self.q.greedy_epsilon(context.state, epsilon=None)
        self.q.bellman_update(context, self.config.lr, self.config.gamma)
