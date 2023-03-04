from collections import defaultdict
from random import randrange, random
from abc import ABC, abstractmethod

from context import Context
from utils import argmax

class AbstractStrategy(ABC):
    """Base class for handling the update algorithm
       using the strategy design pattern"""

    @abstractmethod
    def action(self, context: Context):
        """Generate the next action based on the current context"""
        raise NotImplementedError

    @abstractmethod
    def update(self, context: Context):
        """Update the current model based on the given context"""
        raise NotImplementedError

class QFunction:
    def __init__(self, config):
        self.action_space_shape  = config.action_space_shape
        self.q = defaultdict(lambda : [0.] * self.action_space_shape)

        self.sample = config.env.action_space.sample

    def greedy_epsilon(self, state, epsilon=None):
        # print(f'{epsilon=}')

        if epsilon != None and random() < epsilon:
            # explore
            # print('EXPLORE')
            return self.sample()
            # return randrange(0, self.action_space_shape)
        else:
            # exploit
            # print('EXPLOIT')
            return argmax(self.q[state])

    def bellman_update(self, context, lr, gamma):
        q_current = self.get(context.state, context.action)
        q_future = self.get(context.next_state, context.next_action)
        td_error = context.reward + gamma * q_future - q_current

        new_value = q_current + lr * td_error

        # print(f'{lr=}')
        # print(f'{str(context)=}')
        # print(f'update={context.reward + gamma * q_future}, pred={q_current}, {td_error=}')
        # print(self.get(context.state, context.action))
        self.update(context.state, context.action, new_value)
        # print(self.get(context.state, context.action))

    def get(self, state, action_index) -> float:
        """Override for getting the q value for a
           given state and an action's index """
        actions = self.q[state]
        return actions[action_index]

    def update(self, state, action_index, new_value):
        """Override for getting the q value for a
           given state and an action's index """
        actions = self.q[state]
        actions[action_index] = new_value

class SARSA(AbstractStrategy):
    """SARSA implementation"""

    def __init__(self, config):
        self.config = config
        self.q = QFunction(config)

    def action(self, state) -> int:
        return self.q.greedy_epsilon(state, epsilon=self.config.epsilon)

    def update(self, context: Context):
        context.next_action = self.q.greedy_epsilon(context.state, epsilon=self.config.epsilon)
        self.q.bellman_update(context, self.config.lr, self.config.gamma)

class QLearner(AbstractStrategy):
    """Q-learning implementation"""

    def __init__(self, config):
        self.config = config
        self.q = QFunction(config)

    def action(self, state) -> int:
        return self.q.greedy_epsilon(state, epsilon=self.config.epsilon)

    def update(self, context: Context):
        context.next_action = self.q.greedy_epsilon(context.state, epsilon=None)
        self.q.bellman_update(context, self.config.lr, self.config.gamma)

class Agent:
    """Class representing an agent"""

    def __init__(self, learner: AbstractStrategy):
        self.learner = learner

    def action(self, context: Context) -> int:
        return self.learner.action(context)

    def update(self, context: Context):
        self.learner.update(context)
