import argparse

default_env = 'Blackjack-v1'

def train_arguments():
    parser = argparse.ArgumentParser(description='Trains a memory transformer')
    parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--learning-rate', required=True, type=float)
    parser.add_argument('--env', default=default_env)
    parser.add_argument('--qlearner', action='store_true')
    parser.add_argument('--sarsa', action='store_true')
    parser.add_argument('--save-path')
    parser.add_argument('--load-path')
    parser.add_argument('--show-policy', action='store_true')
    return parser.parse_args()

def test_arguments():
    parser = argparse.ArgumentParser(description='Trains a memory transformer')
    parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--env', default=default_env)
    parser.add_argument('--qlearner', action='store_true')
    parser.add_argument('--sarsa', action='store_true')
    parser.add_argument('--load-path')
    return parser.parse_args()
