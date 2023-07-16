import gymnasium as gym

import torch

from environment import Environment
from model import Primary, Target

class Evaluator:
    def __init__(self, primary_path: str, target_path: str, n_evals):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.environment = Environment(display_mode=True)

        d_in = self.environment.state_shape
        d_out = self.environment.action_shape

        self.primary = Primary(d_in, d_out).to(device).to(device)
        self.target = Target(d_in, d_out, self.primary).to(device)
        self._load_models(primary_path, target_path)

        self.n_evals = n_evals

    def _load_models(self, primary_path: str, target_path: str):
        self.primary.load_state_dict(torch.load(primary_path))
        self.target.model.load_state_dict(torch.load(target_path))

    def eval(self):
        for epoch_iteration in range(self.n_evals):
            state, epoch_reward, done = self.environment.init()

            while not done:
                state, _, done = self.environment.step(state, epoch_reward, self.primary, epsilon=0.05)

        self.environment.cleanup()

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-path', required=True)
    parser.add_argument('--target-path', required=True)
    args = parser.parse_args()

    primary_path = args.primary_path
    target_path = args.target_path

    if not os.path.exists(primary_path) or not os.path.exists(target_path):
        raise ValueError('Model path(s) is invalid')

    evaluator = Evaluator(primary_path, target_path, n_evals=100)
    evaluator.eval()
