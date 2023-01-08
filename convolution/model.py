import torch
from torch import nn

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.f = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),    # 1x28x28  -> 6x28x28
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),                                 # 6x28x28  -> 6x14x14

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),              # 6x14x14  -> 16x10x10
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),                                 # 16x10x10 -> 16x5x5

            nn.Flatten(start_dim=1),
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),

            nn.Linear(120, 84),
            nn.Sigmoid(),

            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.f(x)
