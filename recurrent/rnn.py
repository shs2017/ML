from torch import nn, Tensor

from config import Config
from recurrent import Recurrent, hidden_only_loss_fn


class RNN(Recurrent):
    def __init__(self, config: Config, optimizer_fn):
        super().__init__(config, optimizer_fn=optimizer_fn)
        self.name = 'RNN'

    def create_model(self):
        return RNNModel(d_in=self.d_in, d_hid=self.d_hid, d_out=self.d_out,
                        n_layers=self.n_layers,
                        vocab_size=self.vocab_size).to(self.device)
        
    def loss_type(self):
        return hidden_only_loss_fn

    def train(self, dataset, n_epochs: int, vocab):
        self.train_helper(dataset, n_epochs, vocab)

    def test(self, dataset, vocab):
        self.test_helper(self.model, dataset, hidden_only_loss_fn, vocab)


class RNNModel(nn.Module):
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
        return RNNCell(d_in=self.d_in, d_hid=self.d_hid,
                       d_out=self.d_in, out_is_logits=False)

    def forward(self,
                input_embedding_indices: Tensor,
                hidden_states: [Tensor]) -> (Tensor, [Tensor]):
        state = self.embedding(input_embedding_indices)

        new_hidden_states = []
        for hidden_cell, hidden_state in zip(self.hidden_cells, hidden_states):
            state, hidden_state = hidden_cell(state, hidden_state)
            new_hidden_states.append(hidden_state)

        output_state = self.output_layer(state)

        return output_state, new_hidden_states


class RNNCell(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int, out_is_logits: bool):
        super().__init__()

        self.hidden_to_hidden = nn.Linear(d_hid, d_hid)
        self.state_to_hidden = nn.Linear(d_in, d_hid)
        self.hidden_to_output = nn.Linear(d_hid, d_out)

        self.sigma_hidden = nn.Tanh()
        self.sigma_out = nn.Identity() if out_is_logits else nn.Tanh()

    def forward(self, state: Tensor, hidden_state: Tensor) -> (Tensor, Tensor):
        hidden_from_hidden = self.hidden_to_hidden(hidden_state)
        hidden_from_state = self.state_to_hidden(state)

        new_hidden_logits = hidden_from_hidden + hidden_from_state
        new_hidden_state = self.sigma_hidden(new_hidden_logits)

        output = self.sigma_out(self.hidden_to_output(new_hidden_state))

        return output, new_hidden_state
