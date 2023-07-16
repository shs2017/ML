import numpy as np

import torch
from torch import LongTensor, Tensor

from collections import deque, namedtuple
from random import sample

from preprocess import Preprocess

ReplayMemory = namedtuple('ReplayMemory', ['state', 'action', 'reward', 'next_state', 'next_state_mask'])

class ReplayBuffer:
    def __init__(self, max_size: int, device: str):
        self.replay = deque(maxlen = max_size)
        self.preprocess = Preprocess(device=device)

    def add(self, state, action, reward, next_state) -> None:
        replay_memory = ReplayMemory(state = state,
                                     action = action,
                                     reward = reward,
                                     next_state = next_state,
                                     next_state_mask = next_state is not None)
        self.replay.append(replay_memory)

    def sample(self, n_samples):
        samples = sample(self.replay, n_samples)
        samples = ReplayMemory(*zip(*samples))

        state = self.preprocess.states(samples.state)
        reward = self.preprocess.rewards(samples.reward)
        action = self.preprocess.actions(samples.action)
        next_state = self.preprocess.states([next_state for next_state in samples.next_state if next_state is not None])
        next_state_mask = self.preprocess.mask(samples.next_state_mask)

        return ReplayMemory(state=state, action=action, reward=reward, next_state=next_state, next_state_mask=next_state_mask)

    def __len__(self):
        return len(self.replay)
