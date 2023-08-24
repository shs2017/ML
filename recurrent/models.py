import torch

from torch import nn, Tensor


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


class RNN(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int, n_layers: int):
        super().__init__()

        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out

        self.hidden_cells = nn.ModuleList([
            self._create_hidden_cell()
            for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(d_hid, d_out)

    def _create_hidden_cell(self):
        return RNNCell(d_in=self.d_in, d_hid=self.d_hid,
                       d_out=self.d_in, out_is_logits=False)

    def forward(self, state: Tensor, hidden_states: [Tensor]) -> (Tensor,
                                                                  [Tensor]):
        new_hidden_states = []
        for hidden_cell, hidden_state in zip(self.hidden_cells, hidden_states):
            state, hidden_state = hidden_cell(state, hidden_state)
            new_hidden_states.append(hidden_state)

        output_state = self.output_layer(state)

        return output_state, new_hidden_states


class Gate(nn.Module):
    def __init__(self, d_in: int, d_hidden, d_out: int):
        super().__init__()

        self.W_input_to_hidden = nn.Linear(d_in, d_out)
        self.W_hidden_to_hidden = nn.Linear(d_hidden, d_out)

    def forward(self, input_state, hidden_state):
        input_hidden_logits = self.W_input_to_hidden(input_state)
        hidden_hidden_logits = self.W_hidden_to_hidden(hidden_state)
        return torch.sigmoid(hidden_hidden_logits + input_hidden_logits)


class StateCombiner(nn.Module):
    def __init__(self, d_in1, d_in2, d_out):
        super().__init__()
        self.proj1 = nn.Linear(d_in1, d_out)
        self.proj2 = nn.Linear(d_in2, d_out)

    def forward(self, state1: Tensor, state2: Tensor) -> Tensor:
        return self.proj1(state1) + self.proj2(state2)


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


class LSTM(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int, n_layers: int):
        super().__init__()

        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out

        self.hidden_cells = nn.ModuleList([
            self._create_hidden_cell()
            for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(d_hid, d_out)

    def _create_hidden_cell(self):
        return LSTMCell(d_in=self.d_in, d_hid=self.d_hid, d_out=self.d_hid)

    def forward(self,
                state: Tensor,
                hidden_states: [Tensor],
                previous_cells: [Tensor]) -> (Tensor, [Tensor]):
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


class GRU(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int, n_layers: int):
        super().__init__()

        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out

        self.hidden_cells = nn.ModuleList([
            self._create_hidden_cell()
            for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(d_hid, d_out)

    def _create_hidden_cell(self):
        return GRUCell(d_in=self.d_in, d_hid=self.d_hid, d_out=self.d_hid)

    def forward(self,
                state: Tensor,
                hidden_states: [Tensor]) -> (Tensor, [Tensor]):
        next_hidden_states = []
        for hidden_cell, hidden_state in zip(self.hidden_cells, hidden_states):
            hidden_state = hidden_cell(state, hidden_state)
            next_hidden_states.append(hidden_state)

        output_state = self.output_layer(hidden_state)
        return output_state, next_hidden_states
