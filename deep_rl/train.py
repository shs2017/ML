import gymnasium as gym

import torch
from torch import nn, Tensor
from torch.optim import AdamW

from config import get_config
from environment import Environment
from evaluator import Evaluator
from model import Primary, Target
from replay import ReplayBuffer
from schedule import EpsilonScheduler

config = get_config()
torch.manual_seed(seed=config.seed)

class Trainer:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.batch_size = config.batch_size
        self.replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size, device=device)

        self.environment = Environment(replay_buffer=self.replay_buffer)
        self.scheduler = EpsilonScheduler()

        in_channels = config.frames_per_state
        d_out = self.environment.action_shape

        self.primary = Primary(in_channels, d_out).to(device).to(device)
        self.target = Target(in_channels, d_out, self.primary).to(device)

        self.optimizer = AdamW(self.primary.parameters(),
                               lr=config.lr,
                               betas=config.betas,
                               amsgrad=True)

        self.criterion = nn.HuberLoss()
        self.train_mode = True

    def train(self):
        reward_log = []

        for epoch_iteration in range(config.n_episodes):
            epoch_reward = self.epoch_iteration()
            reward_log.append(epoch_reward)

            if epoch_iteration % config.log_interval == 0 and epoch_iteration != 0:
                avg_reward = sum(reward_log) / len(reward_log)
                print(f'Iteration {"#" + str(epoch_iteration):>5s}, avg_reward = {avg_reward:4f}, epsilon = {self.scheduler.epsilon:4f}')
                reward_log = []
                self.save_model(epoch_iteration)

        self.environment.cleanup()

    def epoch_iteration(self) -> Tensor:
        state, epoch_reward, done = self.environment.init()

        while not done:
            state, epoch_reward, done = self.environment.step(state, epoch_reward, self.primary, self.scheduler.epsilon)

            if len(self.replay_buffer) >= config.batch_size:
                self.update_step()

        return epoch_reward

    def update_step(self) -> None:
        self.scheduler.step()

        batch = self.replay_buffer.sample(n_samples=self.batch_size)
        loss = self.compute_loss(batch)

        self.update_models(loss)

    def compute_loss(self, batch) -> Tensor:
        primary_rewards = self.primary(batch)
        target_rewards = self.target.estimate_next_state_reward(batch)
        return self.criterion(primary_rewards, target_rewards)

    def update_models(self, loss) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.primary.parameters(), config.clip_value)
        self.optimizer.step()
        self.target.ema_update(self.primary)

    def load_model(self, primary_path: str, target_path: str):
        self.primary.load_state_dict(torch.load(primary_path))
        self.target.model.load_state_dict(torch.load(target_path))

    def save_model(self, epoch_iteration: int) -> None:
        torch.save(self.primary.state_dict(), f'models/primary_iteration_{str(epoch_iteration)}.pt')
        torch.save(self.target.model.state_dict(), f'models/target_iteration_{str(epoch_iteration)}.pt')

if __name__ == '__main__':
    import os

    model_path = './models'
    if not(os.path.exists(model_path) and os.path.isdir(model_path)):
        os.makedirs(model_path)

    trainer = Trainer()
    trainer.train()
