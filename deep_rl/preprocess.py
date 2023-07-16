import torch
from torch import Tensor
from torchvision.transforms.functional import resize

import numpy as np

from einops import rearrange

class Preprocess:
    def __init__(self, device: str, channels: int = 1):
        self.device = device
        self.channels = channels

    def states(self, l):
        l = np.array(l)
        return Tensor(l).to(self.device)

    def rewards(self, l):
        return Tensor(l).to(self.device)

    def actions(self, l):
        return Tensor(l).long().unsqueeze(1).to(self.device)

    def mask(self, l):
        return Tensor(l).bool().to(self.device)
