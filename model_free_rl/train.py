import gymnasium as gym

from model import Agent, SARSA
from context import Context
from config import Config

class Trainer:
    """Agent trainer"""

    def __init__(self, env, config: Config):
        self.env = env
        self.config = config
        self.rewards = []

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

            agent.action(context)
            agent.update(context)

            self.rewards.append(reward)

            done = terminated or truncated
            self._log(epoch, context)

    def reward_logs(self):
        jump = 1_000

        for i in range(0, len(self.rewards), jump):
            begin = i
            size = min(len(self.rewards), jump)
            end = begin + size

            print(sum(self.rewards[begin : end]) / size)


    def _log(self, epoch, context):
        if not self.config.debug:
            return

        print(f'Epoch #{epoch}')
        print(f'Context: {context}')

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')

    config = Config(action_space_shape = env.action_space.n,
                    epochs = 10_000,
                    debug = False,
                    gamma = 0.9,
                    lr = 0.5,
                    epsilon = 0.01)
    learner = SARSA(config)
    agent = Agent(learner=learner)

    trainer = Trainer(env, config)
    trainer.fit(agent)

    trainer.reward_logs()
