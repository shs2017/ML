import gymnasium as gym

from model import Agent, SARSA, QLearner
from context import Context
from config import build_config, get_config
from logger import RewardLogger

from time import sleep

class Trainer:
    """Agent trainer"""

    def __init__(self, train_env, test_env):
        self.train_env = train_env
        self.test_env = test_env

        self.config = get_config()
        self.logger = RewardLogger()

    def fit(self, agent: Agent):
        """Fit the agent against the environment"""

        agent.train_mode()
        for epoch in range(self.config.train_epochs):
            self.iteration(self.train_env, epoch, agent, train=True)
            agent.end_epoch()

        train_env.close()

    def test(self, agent: Agent):
        """Test the agent against the environment"""

        agent.test_mode()
        for epoch in range(self.config.test_epochs):
            self.iteration(self.test_env, epoch, agent, train=False)
            agent.end_epoch()

        test_env.close()

    def iteration(self, env, epoch: int, agent: Agent, train: bool):
        done = False
        state, _ = env.reset()

        while not done:
            # Action
            action = agent.action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            context = Context(action, state, next_state, reward)

            # Update
            agent.update(context)
            if not train:
                print(context)


            done = terminated or truncated
            state = context.next_state

            self.logger.update(reward)

        self.logger.log(epoch)

if __name__ == '__main__':
    train_env = gym.make('Blackjack-v1')
    test_env = gym.make('Blackjack-v1')
    build_config(train_env.action_space.n)

    print('Training Q Learning Agent')
    q_agent = Agent(learner=QLearner())
    trainer = Trainer(train_env, test_env)
    trainer.fit(q_agent)

    print('Training SARSA Learning Agent')
    s_agent = Agent(learner=SARSA())
    trainer = Trainer(train_env, test_env)
    trainer.fit(s_agent)

    print('Testing Q Learning Agent')
    trainer.test(q_agent)

    print('Testing SARSA Agent')
    trainer.test(s_agent)
