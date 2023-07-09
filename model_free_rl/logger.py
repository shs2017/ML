from config import get_config

config = get_config()

class RunningRewards:
    def __init__(self):
        self.i = 0
        self.running_rewards = []

        self.epsilon = config.initial_epsilon
        self.log_interval = config.log_interval

    def add_reward(self, reward):
        self.running_rewards.append(reward)
        self.i += 1

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon

    def log(self, pbar):
        n_logs = len(self.running_rewards)
        if n_logs == self.log_interval:
            avg_reward = sum(self.running_rewards) / n_logs

            description = f'Training (Iteration: {self.i:6d}, avg_reward: {avg_reward:4f}, epsilon: {self.epsilon:4f})'
            pbar.set_description(description)

            self.running_rewards = []

