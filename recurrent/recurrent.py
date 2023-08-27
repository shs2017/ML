import torch
import torch.nn.functional as F

from torch import nn, Tensor

from tqdm import tqdm

from config import Config

from abc import ABC, abstractmethod


class Recurrent(ABC):
    def __init__(self, config: Config, optimizer_fn):
        self.d_in = config.d_in
        self.d_hid = config.d_hid
        self.d_out = config.d_out
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.log_interval = config.log_interval

        self.device = config.device
        self.model = self.create_model()
        self.optimizer = optimizer_fn(self.model.parameters())

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
        losses = []
        iterations = 0
        for epoch in range(n_epochs):
            pbar = tqdm(dataset)
            for x, y in pbar:
                loss, outputs = self.calculate_loss(x, y)

                self.model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                iterations += 1                    
                losses.append(loss)
                if iterations % self.log_interval == 0:
                    avg_loss = sum(losses) / len(losses)
                    epoch_str = f'Epoch {epoch + 1} / {n_epochs}'
                    loss_str = f'Average Loss = {avg_loss}'
                    description = f'{epoch_str} | {loss_str}'
                    pbar.set_description(description)
                    losses = []

            predicted_output = vocab.lookup_tokens(outputs)
            sentence = ' '.join(predicted_output)
            print(f'Sample Output: {sentence}')

    def test_helper(self, dataset, vocab):
        with torch.no_grad():
            self.test_helper_body(dataset, vocab)

    def test_helper_body(self, dataset, vocab):
        average_loss = 0.
        i = 0
        pbar = tqdm(dataset)
        for data in pbar:
            x, y = data
            loss, outputs = self.calculate_loss(x, y)

            if i == 0:
                average_loss = loss
            else:
                average_loss = average_loss * ((i - 1) / i) + (loss / i)

            pbar.set_description(f'Loss: {average_loss}')
            i += 1

        predicted_output = vocab.lookup_tokens(outputs)
        print(f'Example output: {predicted_output}')


def hidden_only_loss_fn(model, x: Tensor, y: Tensor, d_in: int, n_layers: int,
                        create_initial_hidden_state_fn) -> (Tensor, Tensor):
    hidden_state = create_initial_hidden_state_fn()

    loss = 0.
    batch_size = x.size(0)
    seq_len = x.size(1)

    outputs = []
    for i in range(seq_len):
        output, hidden_state = model(x[:, i], hidden_state)
        # sum is used for reduction as it is averaged later
        loss += F.cross_entropy(output, y[:, i], reduction='sum')
        outputs.append(logits_to_text(output))

    loss = loss / (seq_len * batch_size)
    return loss, outputs


def hidden_and_cell_loss_fn(model, x, y, d_in: int, n_layers,
                            create_initial_hidden_state_fn) -> (Tensor,
                                                                Tensor):
    cell = create_initial_hidden_state_fn()
    hidden_state = create_initial_hidden_state_fn()

    loss = 0.
    batch_size = x.size(0)
    seq_len = x.size(1)

    outputs = []
    for i in range(seq_len):
        output, hidden_state, cell = model(x[:, i], hidden_state, cell)
        # sum is used for reduction as it is averaged later
        loss += F.cross_entropy(output, y[:, i], reduction='sum')
        outputs.append(logits_to_text(output))

    loss = loss / (seq_len * batch_size)
    return loss, outputs


def logits_to_text(logits: Tensor) -> int:
    predicted_output = logits.argmax(dim=-1)
    return predicted_output[0].item()
