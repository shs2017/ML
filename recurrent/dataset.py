import torch
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader

X = Tensor([[0], [1], [2], [3], [2], [1], [0]]).to(torch.long)
Y = F.one_hot(torch.arange(6, -1, -1), num_classes=10).to(torch.float)
