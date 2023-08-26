import torch

from torch import nn, Tensor


class Gate(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()

        self.W_input_to_hidden = nn.Linear(d_in, d_out)
        self.W_hidden_to_hidden = nn.Linear(d_hidden, d_out)

    def forward(self, input_state: Tensor, hidden_state: Tensor) -> Tensor:
        input_hidden_logits = self.W_input_to_hidden(input_state)
        hidden_hidden_logits = self.W_hidden_to_hidden(hidden_state)
        return torch.sigmoid(hidden_hidden_logits + input_hidden_logits)


class StateCombiner(nn.Module):
    def __init__(self, d_in1: int, d_in2: int, d_out: int):
        super().__init__()
        self.proj1 = nn.Linear(d_in1, d_out)
        self.proj2 = nn.Linear(d_in2, d_out)

    def forward(self, state1: Tensor, state2: Tensor) -> Tensor:
        return self.proj1(state1) + self.proj2(state2)
