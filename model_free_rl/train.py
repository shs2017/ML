import gymnasium as gym

import random
import numpy as np

from tqdm import tqdm

from cli import train_arguments, default_env
from config import get_config
from logger import RunningRewards
from plot import plot_blackjack_policy
from agent import Agent, QLearner, SARSA

config = get_config()
random.seed(a=config.seed)

class Trainer:
    def __init__(self, environment_name: str, agent_type: Agent, epochs: int, lr: float):
        self.running_rewards = RunningRewards()
        self.epsilon = config.initial_epsilon
        self.env = gym.make(environment_name)
        self.agent = agent_type(action_shape=self.env.action_space.n,
                                observation_shape=self.env.observation_space,
                                lr=lr)
        self.epochs = epochs

        self.state = None
        self.next_state = None

    def train(self):
        print(f'Using the {self.agent} agent')
        self.state, _ = self.env.reset(seed=config.seed)

        pbar = tqdm(range(self.epochs))
        for i in pbar:
            self.train_iteration()
            self.running_rewards.log(pbar)

        self.env.close()

    def train_iteration(self):
        action = self.agent.select_action(self.state, epsilon=self.epsilon)
        next_state, reward, terminated, truncated, _ = self.env.step(action)

        self.agent.update(self.state, action, reward, next_state, self.epsilon)
        self.state = next_state

        self.running_rewards.add_reward(reward)
        self.running_rewards.update_epsilon(self.epsilon)

        if terminated or truncated:
            self.state, _ = self.env.reset()
            self.epsilon_schedule_step()
            self.agent.reset_state()

    def epsilon_schedule_step(self):
        decreased_epsilon = self.epsilon - self.epsilon_iteration_decrease
        self.epsilon = max(config.minimum_epsilon, decreased_epsilon)

    @property
    def epsilon_iteration_decrease(self):
        return 3 * (config.initial_epsilon - config.minimum_epsilon) / (self.epochs)

def get_model_types(args):
    if args.qlearner and args.sarsa:
        return [QLearner, SARSA]
    elif args.qlearner:
        return [QLearner]
    elif args.sarsa:
        return [SARSA]
    elif not args.qlearner and not args.sarsa:
        raise ValueError('A model type must be specified')

    raise ValueError('Model is not supported')
    

if __name__ == '__main__':
    args = train_arguments()

    if args.env != default_env:
        raise ValueError(f'Only the "{default_env}" environment is currently supported')

    model_types = get_model_types(args)
    if 1 < len(model_types) and (args.load_path or args.save_path):
        raise ValueError('Support for saving/loading multiple models during ' + \
                         'the same training run is not supported')

    for model_type in model_types:
        trainer = Trainer(args.env, model_type, args.epochs, args.learning_rate)
        if args.load_path:
            trainer.agent.load(args.load_path)

        trainer.train()

        if args.save_path:
            trainer.agent.save(args.save_path)

        if args.show_policy:
            policy = trainer.agent.get_policy()
            plot_blackjack_policy(policy)
