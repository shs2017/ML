from torch import nn

from itertools import chain

def default(x, default_value):
    return x if x is not None else default_value

class SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        layers = []
        sigma = default(config.activation, nn.Sigmoid)

        # Input Layer
        layers.append(nn.Linear(config.d_in, config.d_hid))
        layers.append(sigma())

        # # Hidden Layers
        for _ in range(config.n_hid - 1):
            layers.append(nn.Linear(config.d_hid, config.d_hid))
            layers.append(sigma())

        # Output Layer
        layers.append(nn.Linear(config.d_hid, config.d_out))

        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)
