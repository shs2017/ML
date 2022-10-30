from torch import nn

from itertools import chain

def default(x, default_value):
    return x if x is not None else default_value

class SimpleMLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, n_hid=1, sigma=None):
        super().__init__()

        assert d_in > 0 and d_hid > 0 and d_out > 0 and n_hid > 0

        sigma = default(sigma, nn.ReLU)
        input_layer = [nn.Linear(d_in, d_hid), sigma()]
        hidden_layers = chain(
            *[
                (nn.Linear(d_hid, d_hid), sigma()) for _ in range(n_hid)
            ])
        output_layer = [nn.Linear(d_hid, d_out), sigma()]

        self.f = nn.Sequential(*input_layer, *hidden_layers, *output_layer)

    def forward(self, x):
        return self.f(x)
