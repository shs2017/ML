from models import RNNModel, LSTMModel, GRUModel
from helpers import train_helper, test_helper, \
    hidden_only_loss_fn, hidden_and_cell_loss_fn


class RNN:
    def __init__(self, d_in: int, d_hid: int, d_out: int,
                 n_layers: int, vocab_size: int, optimizer_fn,
                 device: str):
        model = RNNModel(d_in=d_in, d_hid=d_hid, d_out=d_out,
                         n_layers=n_layers,
                         vocab_size=vocab_size).to(device)
        self.name = 'RNN'

        self.optimizer = optimizer_fn(model.parameters(), lr=1e-2)
        self.model = model
        self.device = device

        self.d_in = d_in
        self.n_layers = n_layers

    def train(self, dataset, n_epochs: int, show_output: bool, vocab):
        train_helper(self.model, dataset, self.d_in, n_epochs, self.optimizer,
                     hidden_only_loss_fn, show_output,
                     self.n_layers, vocab, self.device)

    def test(self, dataset, n_layers: int, vocab):
        test_helper(self.model, dataset, self.d_in, hidden_only_loss_fn,
                    self.n_layers, vocab, self.device)


class LSTM:
    def __init__(self, d_in: int, d_hid: int, d_out: int,
                 n_layers: int, vocab_size: int, optimizer_fn,
                 device: str):
        self.name = 'LSTM'

        model = LSTMModel(d_in=d_in, d_hid=d_hid, d_out=d_out,
                          n_layers=n_layers,
                          vocab_size=vocab_size).to(device)
        self.optimizer = optimizer_fn(model.parameters(), lr=1e-2)
        self.model = model
        self.device = device

        self.d_in = d_in
        self.n_layers = n_layers

    def train(self, dataset, n_epochs: int, show_output: bool, vocab):
        train_helper(self.model, dataset, self.d_in, n_epochs, self.optimizer,
                     hidden_and_cell_loss_fn, show_output,
                     self.n_layers, vocab, self.device)

    def test(self, dataset, n_layers: int, vocab):
        test_helper(self.model, dataset, self.d_in, hidden_and_cell_loss_fn,
                    self.n_layers, vocab, self.device)


class GRU:
    def __init__(self, d_in: int, d_hid: int, d_out: int,
                 n_layers: int, vocab_size: int, optimizer_fn,
                 device: str):
        self.name = 'GRU'

        model = GRUModel(d_in=d_in, d_hid=d_hid, d_out=d_out,
                         n_layers=n_layers,
                         vocab_size=vocab_size).to(device)
        self.optimizer = optimizer_fn(model.parameters(), lr=1e-2)
        self.model = model
        self.device = device

        self.d_in = d_in
        self.n_layers = n_layers

    def train(self, dataset, n_epochs: int, show_output: bool, vocab):
        train_helper(self.model, dataset, self.d_in, n_epochs, self.optimizer,
                     hidden_only_loss_fn, show_output,
                     self.n_layers, vocab, self.device)

    def test(self, dataset, n_layers: int, vocab):
        test_helper(self.model, dataset, self.d_in, hidden_only_loss_fn,
                    self.n_layers, vocab, self.device)
