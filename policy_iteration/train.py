import gymnasium as gym

from model import Agent, PolicyIteration
from config import Config

class Context:
    """SARS context

       S: state t
       A: action
       R: reward
       S: state t + 1
    """

    def __init__(self, action, state, next_state, reward):
        self.action = action
        self.state = state
        self.next_state = next_state
        self.reward =reward

    def __str__(self):
        print('(' + \
              f'action={self.action}, ' + \
              f'state={self.state}, ' + \
              f'next_state={self.next_state}, ' + \
              f'reward={self.reward}, ' + \
              ')')

class Trainer:
    """Agent trainer"""

    def __init__(self, env, config: Config):
        self.env = env
        self.config = config

    def fit(self, agent: Agent):
        """Fit the agent against the environment"""
        for epoch in range(self.config.epochs):
            self._iteration(epoch, agent)

        self.env.close()

    def _iteration(self, epoch: int, agent: Agent):
        done = False
        state, _ = self.env.reset()

        while not done:
            action = self.env.action_space.sample()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            context = Context(action, state, next_state, reward)

            done = terminated or truncated
            self._log(epoch, context)

    def _log(self, epoch, context):
        if not self.config.debug:
            return

        print(f'Epoch #{epoch}')
        print(f'Context: {context}')

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    agent = Agent(learner=PolicyIteration())
    config = Config(action_space_shape = env.action_space.n,
                    epochs = 1,
                    debug = False)
    trainer = Trainer(env, config)
    trainer.fit(agent)
