import torch

from torch import Tensor
import torch.nn.functional as F


def create_initial_hidden_state(d_in, n_layers: int, device: str):
    initial_state = torch.ones(d_in).to(device)
    return [initial_state.clone() for _ in range(n_layers)]


def train_helper(model, dataset, d_in: int, n_epochs: int, optimizer,
                 loss_fn, n_layers: int, show_output: bool,
                 vocab, device: str):
    iterations = 0
    for _ in range(n_epochs):
        for x, y in dataset:
            loss, outputs = loss_fn(model, x, y, d_in, n_layers, show_output,
                                    device)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            iterations += 1
            if iterations % 10 == 0:
                print(f'Loss = {loss.item()}')

                if show_output:
                    predicted_output = vocab.lookup_tokens(outputs)
                    sentence = ' '.join(predicted_output)
                    print(f'{sentence}')


def test_helper(model, dataset, d_int: int, loss_fn, n_layers: int,
                vocab, device: str):
    for x, _ in dataset:
        _, outputs = loss_fn(model, x, x, d_in, n_layers, show_output=True,
                             device=device)
        predicted_output = vocab.lookup_tokens(outputs)
        print(f'{predicted_output}')


def hidden_only_loss_fn(model, x: Tensor, y: Tensor, d_in: int, n_layers: int,
                        show_output: bool, device: str):
    hidden_state = create_initial_hidden_state(d_in, n_layers, device)

    loss = 0.
    seq_len = x.size(1)

    outputs = []
    for i in range(seq_len):
        output, hidden_state = model(x[i], hidden_state)
        loss += F.cross_entropy(output, y[i])

        if show_output:
            predicted_output = output.argmax(dim=-1)
            outputs.append(predicted_output[0].item())

    return loss, outputs


def hidden_and_cell_loss_fn(model, x, y, d_in: int, n_layers, show_output: bool,
                            device: str):
    cell = create_initial_hidden_state(d_in, n_layers, device)
    hidden_state = create_initial_hidden_state(d_in, n_layers, device)

    loss = 0.
    seq_len = x.size(1)

    outputs = []
    for i in range(seq_len):
        output, hidden_state, cell = model(x[:, i], hidden_state, cell)
        loss += F.cross_entropy(output, y[:, i])

        if show_output:
            predicted_output = output.argmax(dim=-1)
            outputs.append(predicted_output[0].item())

    return loss, outputs
