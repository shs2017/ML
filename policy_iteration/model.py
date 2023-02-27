from abc import ABC, abstractmethod

class AbstractStrategy(ABC):
    """Base class for handling the update algorithm
       using the strategy design pattern"""

    @abstractmethod
    def action(self):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

class PolicyIteration(AbstractStrategy):
    """Policy iteration implementation"""

    def __init__(config):
        pass

    def action(self):
        pass

    def update(self):
        pass

class Agent:
    """Class representing an agent"""

    def __init__(self, learner: AbstractStrategy):
        self.learner = learner

    def action(self):
        self.learner.action()

    def update(self):
        self.learner.update()
