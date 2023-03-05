from collections import deque

from config import get_config

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
