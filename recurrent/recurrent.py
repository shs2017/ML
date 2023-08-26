import torch
import torch.nn.functional as F

from torch import Tensor

from abc import ABC, abstractmethod


class Recurrent(ABC):
    def __init__(self, d_in: int, d_hid: int, d_out: int, vocab_size: int,
                 n_layers: int, optimizer_fn, device: str):
        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.device = device
        self.model = self.create_model()
        self.optimizer = optimizer_fn(self.model.parameters(), lr=1e-2)

    @abstractmethod
    def create_model(self):
        ...

    @abstractmethod
    def loss_type(self):
        ...

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tensor:
        loss_fn = self.loss_type()

        return loss_fn(self.model, x, y, self.d_in,
                       self.n_layers,
                       self.create_initial_hidden_state)

    def create_initial_hidden_state(self):
        initial_state = torch.ones(self.d_in).to(self.device)
        return [initial_state.clone() for _ in range(self.n_layers)]

    def train_helper(self, dataset, n_epochs: int, vocab):
        iterations = 0
        for _ in range(n_epochs):
            for x, y in dataset:
                loss, outputs = self.calculate_loss(x, y)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                iterations += 1
                if iterations % 10 == 0:
                    print(f'Loss = {loss.item()}')

                    predicted_output = vocab.lookup_tokens(outputs)
                    sentence = ' '.join(predicted_output)
                    print(f'{sentence}')

    def test_helper(self, dataset, vocab):
        for x, _ in dataset:
            _, outputs = self.calculate_loss(self.model, x, x, d_in,
                                             self.n_layers, self.device)
            predicted_output = vocab.lookup_tokens(outputs)
            print(f'{predicted_output}')


def hidden_only_loss_fn(model, x: Tensor, y: Tensor, d_in: int, n_layers: int,
                        create_initial_hidden_state_fn) -> (Tensor, Tensor):
    hidden_state = create_initial_hidden_state_fn()

    loss = 0.
    seq_len = x.size(1)

    outputs = []
    for i in range(seq_len):
        output, hidden_state = model(x[i], hidden_state)
        loss += F.cross_entropy(output, y[i])

        predicted_output = output.argmax(dim=-1)
        outputs.append(logits_to_text(output))

    return loss, outputs


def hidden_and_cell_loss_fn(model, x, y, d_in: int, n_layers,
                            create_initial_hidden_state_fn) -> (Tensor,
                                                                Tensor):
    cell = create_initial_hidden_state_fn()
    hidden_state = create_initial_hidden_state_fn()

    loss = 0.
    seq_len = x.size(1)

    outputs = []
    for i in range(seq_len):
        output, hidden_state, cell = model(x[:, i], hidden_state, cell)
        loss += F.cross_entropy(output, y[:, i])
        outputs.append(logits_to_text(output))

    return loss, outputs


def logits_to_text(logits: Tensor) -> int:
    predicted_output = logits.argmax(dim=-1)
    return predicted_output[0].item()
