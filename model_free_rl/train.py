import gymnasium as gym

from time import sleep
from typing import List

from model import Agent, SARSA, QLearner
from context import Context
from config import get_config
from logger import RewardLogger, graph_policy_maps

config = get_config()

class Trainer:
    """Agent trainer"""

    def __init__(self, train_env, test_env):
        self.train_env = train_env
        self.test_env = test_env
        self.logger = RewardLogger()

    def fit(self, agent: Agent):
        """Fit the agent against the environment"""

        agent.train_mode()
        for epoch in range(config.train_epochs):
            self.iteration(self.train_env, epoch, agent, train=True)
            agent.end_epoch()

        train_env.close()

    def test(self, agent: Agent):
        """Test the agent against the environment"""

        agent.test_mode()
        for epoch in range(config.test_epochs):
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
            context = Context(action, state, next_state, reward, terminated=terminated)

            # Update
            agent.update(context)

            if not train:
                print(context)

            done = terminated or truncated
            state = context.next_state

            self.logger.update(reward)

        self.logger.log(epoch)

if __name__ == '__main__':
    train_with_q_learning = True
    train_with_sarsa = False
    should_graph = True

    train_env = gym.make('Blackjack-v1')
    test_env = gym.make('Blackjack-v1')

    action_space_shape = train_env.action_space.n
    agents = [
        ('Q Learning', QLearner(action_space_shape), train_with_q_learning),
        ('SARSA', SARSA(action_space_shape), train_with_sarsa)
    ]

    for agent_name, learner, should_train in agents:
        if not should_train:
            continue

        print(f'Training {agent_name} Agent')
        agent = Agent(learner=learner)
        trainer = Trainer(train_env, test_env)
        trainer.fit(agent)

        print(f'Testing {agent_name} Agent')
        trainer.test(agent)

        if should_graph:
            graph_policy_maps(agent.policy_maps)
