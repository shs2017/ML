import torch

from torch import nn, Tensor

from layers import Gate, StateCombiner
from recurrent import Recurrent, hidden_and_cell_loss_fn


class LSTM(Recurrent):
    def __init__(self, d_in: int, d_hid: int, d_out: int,
                 n_layers: int, vocab_size: int, optimizer_fn,
                 device: str):
        super().__init__(d_in=d_in, d_hid=d_hid, d_out=d_out,
                         vocab_size=vocab_size, n_layers=n_layers,
                         optimizer_fn=optimizer_fn, device=device)

        self.name = 'LSTM'

    def create_model(self):
        return LSTMModel(d_in=self.d_in, d_hid=self.d_hid, d_out=self.d_out,
                         n_layers=self.n_layers,
                         vocab_size=self.vocab_size).to(self.device)

    def loss_type(self):
        return hidden_and_cell_loss_fn

    def train(self, dataset, n_epochs: int, vocab):
        self.train_helper(dataset, n_epochs, vocab)

    def test(self, dataset, vocab):
        self.test_helper(self.model, dataset, hidden_only_loss_fn, vocab)


class LSTMModel(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int,
                 n_layers: int, vocab_size: int):
        super().__init__()

        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out

        self.embedding = nn.Embedding(vocab_size, d_in)

        self.hidden_cells = nn.ModuleList([
            self._create_hidden_cell()
            for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(d_hid, d_out)

    def _create_hidden_cell(self):
        return LSTMCell(d_in=self.d_in, d_hid=self.d_hid, d_out=self.d_hid)

    def forward(self,
                input_embedding_indices: Tensor,
                hidden_states: [Tensor],
                previous_cells: [Tensor]) -> (Tensor, [Tensor]):

        state = self.embedding(input_embedding_indices)

        next_cells = []
        next_hidden_states = []
        for hidden_cell, hidden_state, previous_cell in zip(self.hidden_cells,
                                                            hidden_states,
                                                            previous_cells):
            hidden_state, cell = hidden_cell(state, hidden_state,
                                             previous_cell)
            next_hidden_states.append(hidden_state)
            next_cells.append(hidden_state)

        output_state = self.output_layer(hidden_state)
        return output_state, next_hidden_states, next_cells


class LSTMCell(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int):
        super().__init__()

        self.input_gate_fn = Gate(d_in, d_hid, d_hid)
        self.output_gate_fn = Gate(d_in, d_hid, d_out)
        self.forget_gate_fn = Gate(d_in, d_hid, d_hid)

        self.new_cell_fn = StateCombiner(d_in, d_hid, d_hid)

    def forward(self,
                input_state: Tensor,
                hidden_state: Tensor,
                previous_cell: Tensor) -> (Tensor, [Tensor]):

        input_gate = self.input_gate_fn(input_state, hidden_state)
        output_gate = self.input_gate_fn(input_state, hidden_state)
        forget_gate = self.input_gate_fn(input_state, hidden_state)

        proposed_cell = torch.tanh(self.new_cell_fn(input_state, hidden_state))

        next_cell = forget_gate * previous_cell + input_gate * proposed_cell
        next_hidden_state = output_gate * torch.tanh(next_cell)

        return next_hidden_state, next_cell
