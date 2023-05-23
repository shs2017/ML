from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from config import get_config
from typing import List, Tuple

class RewardLogger:
    def __init__(self):
        self.config = get_config()
        self.rewards = deque()

    def update(self, reward: float):
        self.rewards.append(reward)
        while self.config.log_interval < len(self.rewards):
                self.rewards.popleft()

    def log(self, epoch: int):
        if epoch == 0:
            return

        if self.config.log and (epoch % self.config.log_interval) == 0:
            average_reward = sum(self.rewards) / len(self.rewards)
            print(f'Epoch #{epoch}: Average Reward {average_reward}')


def graph_policy_maps(policy_maps: Tuple[List[List[int]]]) -> None:
    fig, ax = plt.subplots(ncols=len(policy_maps))
    for i, policy_map in enumerate(policy_maps):
        ax[i].imshow(policy_map)
        ax[i].set_xticks(np.arange(1, 21))
        ax[i].set_yticks(np.arange(1, 11))
    plt.show()
