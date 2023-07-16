import torch
import torch.nn.functional as F

from torch import nn, Tensor
from random import random, randrange, seed

from replay import ReplayMemory
from config import get_config

config = get_config()

seed(a=42)

class DDQN:
    def __init__(self, primary, target):
        self.primary = primary
        self.target = target

class Primary(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        self.d_out = d_out
        self.model = ConvModel(d_in, d_out)

        self.gamma = config.gamma

    def select_action(self, state: Tensor, epsilon: float) -> int:
        if random() < epsilon:
            return randrange(self.d_out)

        with torch.no_grad():
            return self.model(state).argmax(dim=-1).item()

    def forward(self, replay_memory: ReplayMemory) -> Tensor:
        predicted_reward = self.model(replay_memory.state).gather(-1, replay_memory.action)
        return predicted_reward.squeeze(-1)

class Target:
    def __init__(self, d_in: int, d_out: int, primary: Primary):
        super().__init__()

        self.model = ConvModel(d_in, d_out)
        self.model.load_state_dict(primary.model.state_dict())

        self.tau = config.tau

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def estimate_next_state_reward(self, replay_memory: ReplayMemory):
        # Count terminal next_states as having a zero reward
        predicted_rewards = torch.zeros_like(replay_memory.reward)

        with torch.no_grad():
            max_reward = self.model(replay_memory.next_state).max(dim=-1).values
            non_terminal_mask = replay_memory.next_state_mask
            predicted_rewards[non_terminal_mask] = max_reward

        return replay_memory.reward + config.gamma * predicted_rewards

    def ema_update(self, other_model):
        state_dict = self.model.state_dict()
        other_state_dict = other_model.model.state_dict()

        for state in state_dict:
            update = self.tau * state_dict[state] + (1 - self.tau) * other_state_dict[state]
            state_dict[state] = update

        self.model.load_state_dict(state_dict)

class ConvModel(nn.Module):
    def __init__(self, in_channels, d_out):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)

        self.mlp = MLP(d_in=1024, d_out=d_out)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))

        x = x.flatten(start_dim=1)

        x = self.mlp(x)

        return x


class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        d_in = d_in
        d_hid = config.d_hid

        self.f = nn.Sequential(nn.Linear(d_in, d_hid),
                               nn.ReLU(),
                               nn.Linear(d_hid, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.f(x)
