from typing import Sequence

from collections import defaultdict
from random import randrange, random
from abc import ABC, abstractmethod

from context import Context

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

def argmax(l: Sequence) -> int:
    """Arg-max implementation for a list"""
    best_index = None
    best_value = None

    for i, val in enumerate(l):
        if best_value is None or best_value < val:
            val = best_value
            best_index = i

    return best_index

class QFunction:
    def __init__(self, config):
        self.q = defaultdict(lambda : [0.] * config.action_space_shape)

    def best_action(self, state) -> int:
        return argmax(self.q[state])

    def __getitem__(self, state_action_index) -> float:
        """Override for getting the q value for a
           given state and an action's index """
        state, action_index = state_action_index

        actions = self.q[state]

        return actions[action_index]

    def __setitem__(self, state_action_index, new_value):
        """Override for getting the q value for a
           given state and an action's index """
        state, action_index = state_action_index

        actions = self.q[state]
        actions[action_index] = new_value

class SARSA(AbstractStrategy):
    """SARSA implementation"""

    def __init__(self, config):
        self.gamma = config.gamma
        self.lr = config.gamma
        self.epsilon = config.epsilon
        self.q = QFunction(config)
        self.action_space_shape = config.action_space_shape

    def action(self, context: Context):
        state = context.state
        action = context.action

        next_state = context.next_state

        if random() < self.epsilon:
            # explore
            context.next_action = randrange(0, self.action_space_shape)
        else:
            # exploit
            context.next_action = self.q.best_action(next_state)

    def update(self, context: Context):
        reward = context.reward

        state = context.state
        action = context.action

        next_state = context.next_state
        next_action = context.next_action

        q_current = self.q[state, action]
        q_future = self.q[next_state, next_action]

        td_error = reward + self.gamma * q_future - q_current
        new_value = q_current + self.lr * td_error

        self.q[state, action] = new_value

class QLearning(AbstractStrategy):
    """Q-Learning implementation"""

    def __init__(self, config):
        pass

    def action(self):
        pass

    def update(self, context: Context):
        pass

class Agent:
    """Class representing an agent"""

    def __init__(self, learner: AbstractStrategy):
        self.learner = learner

    def action(self, context: Context):
        self.learner.action(context)

    def update(self, context: Context):
        self.learner.update(context)
