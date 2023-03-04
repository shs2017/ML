import gymnasium as gym

from model import Agent, SARSA, QLearner
from context import Context
from config import Config

class Trainer:
    """Agent trainer"""

    def __init__(self, env, config: Config):
        self.env = env
        self.config = config
        self.rewards = []

    def fit(self, agent: Agent, train=True):
        """Fit the agent against the environment"""

        for epoch in range(self.config.epochs):
            self._iteration(epoch, agent, train)
            self.config.epsilon = max(self.config.epsilon_min,
                                      self.config.epsilon - self.config.epsilon_step_size)
            if not train:
                print('-' * 80)

        self.env.close()

    def _iteration(self, epoch: int, agent: Agent, train: bool):
        done = False
        state, _ = self.env.reset()

        while not done:
            # Action
            action = agent.action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            context = Context(action, state, next_state, reward)

            # Update
            if train:
                agent.update(context)
            else:
                print(context)

            done = terminated or truncated
            state = context.next_state

            # Logging
            self.rewards.append(reward)

            if len(self.rewards) >= 1_000 and not train:
                print(f'Average Reward {sum(self.rewards) / len(self.rewards)}')
                self.rewards = []

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')

    config = Config(action_space_shape = env.action_space.n,
                    epochs = 100_000,
                    debug = False,
                    gamma = 1,
                    lr = 0.01,
                    epsilon = 1,
                    epsilon_min = 0.1,
                    epsilon_step_size = 0.001,
                    env = env)

    learner_algorithm = QLearner
    print(f'{learner_algorithm=}')
    learner = learner_algorithm(config)
    agent = Agent(learner=learner)

    trainer = Trainer(env, config)
    trainer.fit(agent)

    config.epochs=10_000
    config.epsilon=0
    config.epsilon_step_size=0
    config.epsilon_min=0

    trainer.fit(agent, train=False)
