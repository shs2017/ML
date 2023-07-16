import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

import torch
from torch import Tensor

from config import get_config
from preprocess import Preprocess
from replay import ReplayBuffer

config = get_config()

class Environment:
    def __init__(self, replay_buffer: ReplayBuffer = None, display_mode: bool = False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.replay_buffer = replay_buffer if replay_buffer is not None else None

        render_mode = 'human' if display_mode else None

        env = gym.make(config.env_name, obs_type='grayscale', render_mode=render_mode)
        env = AtariPreprocessing(env)
        env = FrameStack(env, num_stack = config.frames_per_state)
        self.env = env

        self.preprocess = Preprocess(device)

    @property
    def state_shape(self):
        return self.env.observation_space.shape

    @property
    def action_shape(self):
        return self.env.action_space.n

    def init(self):
        epoch_reward = 0.
        done = False
        state, _ = self.env.reset(seed=config.seed)

        return state, epoch_reward, done

    def step(self, state, epoch_reward, model, epsilon):
        states = self.preprocess.states(state).unsqueeze(0)
        action = model.select_action(states, epsilon)
        next_state, reward, terminated, truncated, _ = self.env.step(action)

        done = terminated or truncated

        next_state = next_state if not done else None

        if self.replay_buffer is not None:
            self.replay_buffer.add(state=state, action=action, reward=reward, next_state=next_state)

        return next_state, epoch_reward + reward, done

    def cleanup(self):
        self.env.close()

    def __getattr__(self, attr):
        return self.env.env
