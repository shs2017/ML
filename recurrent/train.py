import torch
import torch.nn.functional as F

from torch.optim import SGD

from dataset import X, Y
from models import RNN, LSTM, GRU


def create_initial_hidden_state(n_layers):
    initial_state = torch.ones(d_in)
    return [initial_state.clone() for _ in range(n_layers)]


def train_rnn(model, X, Y, n_epochs, optimizer):
    n_samples = X.size(0)

    iterations = 0
    for _ in range(n_epochs):
        loss = 0.
        hidden_state = create_initial_hidden_state(n_layers)
        for i in range(n_samples):
            x, y = X[i], Y[i]
            y = y.unsqueeze(0)

            output, hidden_state = model(x, hidden_state)
            loss += F.binary_cross_entropy_with_logits(output, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

    iterations += 1

    if iterations % 100 == 0:
        print(loss)


def train_lstm(model, X, Y, n_epochs, optimizer):
    n_samples = X.size(0)

    iterations = 0
    for _ in range(n_epochs):
        loss = 0.
        cell = create_initial_hidden_state(n_layers)
        hidden_state = create_initial_hidden_state(n_layers)
        for i in range(n_samples):
            x, y = X[i], Y[i]
            y = y.unsqueeze(0)

            output, hidden_state, cell = model(x, hidden_state, cell)
            loss += F.binary_cross_entropy_with_logits(output, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

    iterations += 1

    if iterations % 100 == 0:
        print(loss)


def train_gru(model, X, Y, n_epochs, optimizer):
    n_samples = X.size(0)

    iterations = 0
    for _ in range(n_epochs):
        loss = 0.
        hidden_state = create_initial_hidden_state(n_layers)
        for i in range(n_samples):
            x, y = X[i], Y[i]
            y = y.unsqueeze(0)

            output, hidden_state = model(x, hidden_state)
            loss += F.binary_cross_entropy_with_logits(output, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

    iterations += 1

    if iterations % 100 == 0:
        print(loss)


def test_rnn(model, X):
    n_samples = X.size(0)

    hidden_state = create_initial_hidden_state(n_layers)
    for i in range(n_samples):
        x = X[i]
        output, hidden_state = model(x, hidden_state)
        output = torch.sigmoid(output)
        print(f'{x} -> {torch.argmax(output)}')


def test_lstm(model, X):
    n_samples = X.size(0)

    cell = create_initial_hidden_state(n_layers)
    hidden_state = create_initial_hidden_state(n_layers)
    for i in range(n_samples):
        x = X[i]
        output, hidden_state, cell = model(x, hidden_state, cell)
        output = torch.sigmoid(output)
        print(f'{x} -> {torch.argmax(output)}')


def test_gru(model, X):
    n_samples = X.size(0)

    hidden_state = create_initial_hidden_state(n_layers)
    for i in range(n_samples):
        x = X[i]
        output, hidden_state = model(x, hidden_state)
        output = torch.sigmoid(output)
        print(f'{x} -> {torch.argmax(output)}')


def train_and_test_rnn(d_in: int, d_hid: int, d_out: int, n_layers: int,
                       dataset, vocab_size):
    X, Y = dataset
    rnn_model = RNN(d_in=d_in, d_hid=d_hid, d_out=d_out,
                    n_layers=n_layers, vocab_size=vocab_size)
    rnn_optimizer = SGD(rnn_model.parameters(), lr=1e-1)
    train_rnn(rnn_model, X, Y, n_epochs, rnn_optimizer)
    test_rnn(rnn_model, X)


def train_and_test_lstm(d_in: int, d_hid: int, d_out: int, n_layers: int,
                        dataset, vocab_size):
    X, Y = dataset
    lstm_model = LSTM(d_in=d_in, d_hid=d_hid, d_out=d_out,
                      n_layers=n_layers, vocab_size=vocab_size)
    lstm_optimizer = SGD(lstm_model.parameters(), lr=1e-1)
    train_lstm(lstm_model, X, Y, n_epochs, lstm_optimizer)
    test_lstm(lstm_model, X)


def train_and_test_gru(d_in: int, d_hid: int, d_out: int, n_layers: int,
                       dataset, vocab_size):
    X, Y = dataset
    gru_model = GRU(d_in=d_in, d_hid=d_hid, d_out=d_out,
                    n_layers=n_layers, vocab_size=vocab_size)
    gru_optimizer = SGD(gru_model.parameters(), lr=1e-1)
    train_gru(gru_model, X, Y, n_epochs, gru_optimizer)
    test_gru(gru_model, X)


if __name__ == '__main__':
    n_epochs = 2_000
    n_layers = 3

    d_in = 10
    d_hid = 10
    d_out = 10

    dataset = (X, Y)

    vocab_size = 10

    print('Training and testing an RNN...')
    train_and_test_rnn(d_in=d_in, d_hid=d_hid, d_out=d_out, n_layers=n_layers,
                       dataset=dataset, vocab_size=vocab_size)

    print('Training and testing a LSTM...')
    train_and_test_lstm(d_in=d_in, d_hid=d_hid, d_out=d_out, n_layers=n_layers,
                        dataset=dataset, vocab_size=vocab_size)

    print('Training and testing a GRU...')
    train_and_test_gru(d_in=d_in, d_hid=d_hid, d_out=d_out, n_layers=n_layers,
                       dataset=dataset, vocab_size=vocab_size)
