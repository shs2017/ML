import torch
from torch import nn

DEFAULT_KERNEL_SIZE = 3
DEFAULT_PADDING = 1

class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()

        self.fn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.fn(x)


class Skip(nn.Module):
    def __init__(self, fn, in_channels, scale_factor=1):
        super().__init__()

        out_channels = int(scale_factor * in_channels)

        if scale_factor == 1:
            self.scale = nn.Identity()
        else:
            self.scale = BatchNormConv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=DEFAULT_KERNEL_SIZE,
                                         stride=scale_factor,
                                         padding=1)
        self.fn = fn
        self.sigma = nn.ReLU()

    def forward(self, x):
        return self.sigma(self.fn(x) + self.scale(x))


class Conv2dGroup(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()

        layers = [
            nn.Sequential(
                BatchNormConv2d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=DEFAULT_KERNEL_SIZE,
                                padding=DEFAULT_PADDING),
                nn.ReLU()
            ) for _ in range(num_layers)]

        self.fn = Skip(nn.Sequential(*layers), channels)

    def forward(self, x):
        return self.fn(x)


class NormalBlock(nn.Module):
    def __init__(self, channels, num_groups, layers_per_group=2):
        super().__init__()

        self.fn = nn.Sequential(*[Conv2dGroup(channels, layers_per_group)
                                  for _ in range(num_groups)])

    def forward(self, x):
        return self.fn(x)


class ScaleBlock(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        conv = nn.Sequential(
            BatchNormConv2d(in_channels=channels,
                            out_channels=scale_factor * channels,
                            kernel_size=DEFAULT_KERNEL_SIZE,
                            stride=scale_factor,
                            padding=1),
            BatchNormConv2d(in_channels=scale_factor * channels,
                            out_channels=scale_factor * channels,
                            kernel_size=DEFAULT_KERNEL_SIZE,
                            padding=1)

        )


        self.f = Skip(conv, channels, scale_factor)

    def forward(self, x):
        return self.f(x)


class Block(nn.Module):
    def __init__(self, channels, num_groups, scale_factor=2):
        super().__init__()

        self.fn = nn.Sequential(
            NormalBlock(channels, num_groups),
            ScaleBlock(channels, scale_factor)
        )

    def forward(self, x):
        return self.fn(x)


class Residuals(nn.Module):
    def __init__(self, out_features=257):
        super().__init__()

        self.conv1 = BatchNormConv2d(in_channels=3,
                                     out_channels=64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        channels=64

        layers = []
        layers.append(Block(channels, num_groups=3))
        channels *= 2

        layers.append(Block(channels, num_groups=3))
        channels *= 2

        layers.append(Block(channels, num_groups=5))
        channels *= 2

        layers.append(NormalBlock(channels, num_groups=2))

        self.blocks = nn.Sequential(*layers)

        self.global_average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 257)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.blocks(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x
