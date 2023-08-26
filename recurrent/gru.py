from torch import nn, Tensor

from layers import Gate, StateCombiner
from recurrent import Recurrent, hidden_only_loss_fn


class GRU(Recurrent):
    def __init__(self, d_in: int, d_hid: int, d_out: int,
                 n_layers: int, vocab_size: int, optimizer_fn,
                 device: str):
        super().__init__(d_in=d_in, d_hid=d_hid, d_out=d_out,
                         vocab_size=vocab_size, n_layers=n_layers,
                         optimizer_fn=optimizer_fn, device=device)
        self.name = 'GRU'

    def create_model(self):
        return GRUModel(d_in=self.d_in, d_hid=self.d_hid, d_out=self.d_out,
                        n_layers=self.n_layers,
                        vocab_size=self.vocab_size).to(self.device)

    def loss_type(self):
        return hidden_only_loss_fn

    def train(self, dataset, n_epochs: int, vocab):
        self.train_helper(dataset, n_epochs, vocab)

    def test(self, dataset, vocab):
        self.test_helper(self.model, dataset, hidden_only_loss_fn, vocab)


class GRUModel(nn.Module):
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
        return GRUCell(d_in=self.d_in, d_hid=self.d_hid, d_out=self.d_hid)

    def forward(self,
                input_embedding_indices: Tensor,
                hidden_states: [Tensor]) -> (Tensor, [Tensor]):

        state = self.embedding(input_embedding_indices)

        next_hidden_states = []
        for hidden_cell, hidden_state in zip(self.hidden_cells, hidden_states):
            hidden_state = hidden_cell(state, hidden_state)
            next_hidden_states.append(hidden_state)

        output_state = self.output_layer(hidden_state)
        return output_state, next_hidden_states


class GRUCell(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int):
        super().__init__()
        self.forget_fn = Gate(d_in=d_in, d_hidden=d_hid, d_out=d_hid)
        self.update_fn = Gate(d_in=d_in, d_hidden=d_hid, d_out=d_hid)

        self.proposed_hidden_fn = StateCombiner(d_in, d_hid, d_hid)

    def forward(self, state: Tensor, hidden_state: Tensor) -> (Tensor,
                                                               Tensor):
        forget_gate = self.forget_fn(state, hidden_state)
        update_gate = self.update_fn(state, hidden_state)

        remembered_hidden_state = forget_gate * hidden_state
        proposed_hidden = self.proposed_hidden_fn(state,
                                                  remembered_hidden_state)

        next_hidden = (update_gate) * hidden_state + \
            (1 - update_gate) * proposed_hidden

        return next_hidden
