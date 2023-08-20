from torch import nn, Tensor


class RNNCell(nn.Module):
    def __init__(self, d_hid: int, d_state: int, d_out: int, out_is_logits: bool):
        super().__init__()

        self.hidden_to_hidden = nn.Linear(d_hid, d_hid)
        self.state_to_hidden = nn.Linear(d_state, d_hid)
        self.hidden_to_output = nn.Linear(d_hid, d_out)

        self.sigma_hidden = nn.Tanh()
        self.sigma_out = nn.Identity() if out_is_logits else nn.Tanh()

    def forward(self, state, hidden_state):
        hidden_from_hidden = self.hidden_to_hidden(hidden_state)
        hidden_from_state = self.state_to_hidden(state)

        new_hidden_logits = hidden_from_hidden + hidden_from_state
        new_hidden_state = self.sigma_hidden(new_hidden_logits)

        output = self.sigma_out(self.hidden_to_output(new_hidden_state))

        return output, new_hidden_state


class RNN(nn.Module):
    def __init__(self, d_hid: int, d_state: int, d_out: int, n_layers: int):
        super().__init__()

        self.d_hid = d_hid
        self.d_state = d_state
        self.d_out = d_out

        self.hidden_cells = nn.ModuleList([
            self._create_hidden_cell()
            for _ in range(n_layers - 1)
        ])
        self.output_cell = self._create_output_cell()

    def _create_hidden_cell(self):
        return RNNCell(d_hid=self.d_hid, d_state=self.d_state,
                       d_out=self.d_state, out_is_logits=False)

    def _create_output_cell(self):
        return RNNCell(d_hid=self.d_hid, d_state=self.d_state,
                       d_out=self.d_out, out_is_logits=True)

    def forward(self, state: Tensor, hidden_states: [Tensor]) -> [Tensor]:
        new_hidden_states = []

        inner_states, last_hidden = hidden_states[:-1], hidden_states[-1]

        for hidden_cell, hidden_state in zip(self.hidden_cells, inner_states):
            state, hidden_state = hidden_cell(state, hidden_state)
            new_hidden_states.append(hidden_state)

        output_state, hidden_state = self.output_cell(state, hidden_state)
        new_hidden_states.append(hidden_state)

        return output_state, new_hidden_states
