import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

X = F.one_hot(torch.arange(5), num_classes=10).to(torch.float)
Y = F.one_hot(torch.arange(4, -1, -1), num_classes=10).to(torch.float)
