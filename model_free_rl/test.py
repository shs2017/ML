import gymnasium as gym

import random
import numpy as np
from time import sleep

from tqdm import tqdm

from cli import test_arguments, default_environment
from config import get_config
from logger import RunningRewards
from plot import plot_blackjack_policy
from agent import Agent, QLearner, SARSA

config = get_config()
random.seed(a=config.seed)

class Tester:
    def __init__(self, environment_name: str, agent_type: Agent, epochs: int):
        self.running_rewards = RunningRewards()
        self.env = gym.make(environment_name, render_mode='human')
        self.agent = agent_type(action_shape=self.env.action_space.n,
                                observation_shape=self.env.observation_space,
                                lr=None)
        self.epochs = epochs
        self.state = None
        self.next_state = None

    def test(self):
        print(f'Using the {self.agent} agent')
        self.state, _ = self.env.reset(seed=config.seed)
        sleep(1)

        pbar = tqdm(range(self.epochs))
        for i in pbar:
            self.test_iteration()
            self.running_rewards.log(pbar)
            sleep(1)

        self.env.close()

    def test_iteration(self):
        action = self.agent.select_action(self.state, epsilon=0)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.state = next_state

        if terminated or truncated:
            self.state, _ = self.env.reset()
            self.agent.reset_state()

def get_model_types(args):
    if args.qlearner and args.sarsa:
        raise ValueError('Support for testing multiple models in the same run is not supported')
    elif args.qlearner:
        return QLearner
    elif args.sarsa:
        return SARSA
    elif not args.qlearner and not args.sars:
        raise ValueError('A model type must be specified')

    raise ValueError('Model is not supported')
    

if __name__ == '__main__':
    args = test_arguments()

    if args.env != default_env:
        raise ValueError(f'Only the "{default_env}" environment is currently supported')

    model_type = get_model_types(args)
    tester = Tester(args.env, model_type, args.epochs)
    if not args.load_path:
        raise Value('A model path is required')

    tester.agent.load(args.load_path)
    tester.test()
