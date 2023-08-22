import torch
import torch.nn.functional as F

from torch.optim import SGD

from dataset import X, Y
from models import RNN


rand = torch.rand(10)

n_layers = 3
d_hid = 10
d_state = 10
d_out = 10

# rnn_cell = RNNCell(d_hid=d_hid, d_state=d_state,
#                    d_out=d_out, out_is_logits=False)
# out_cell = rnn_cell(rand, rand)
# print(out_cell)


def create_initial_hidden_state(n_layers):
    initial_state = torch.ones(d_state)
    return [initial_state.clone() for _ in range(n_layers)]

def train(model, X, Y, n_epochs, optimizer):
    n_samples = X.size(0)

    iterations = 0
    for _ in range(n_epochs):
        loss = 0.
        hidden_state = create_initial_hidden_state(n_layers)
        for i in range(n_samples):
            x, y = X[i], Y[i]

            output, hidden_state = model(x, hidden_state)
            loss += F.binary_cross_entropy_with_logits(output, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

    iterations += 1

    if iterations % 100 == 0:
        print(loss)


def test(model, X):
    n_samples = X.size(0)

    hidden_state = create_initial_hidden_state(n_layers)
    for i in range(n_samples):
        x = X[i]
        output, hidden_state = model(x, hidden_state)
        output = torch.sigmoid(output)
        print(f'{torch.argmax(x)} -> {torch.argmax(output)}')


n_epochs = 2_000

model = RNN(d_hid=d_hid, d_state=d_state, d_out=d_out, n_layers=n_layers)
optimizer = SGD(model.parameters(), lr=1e-1)
train(model, X, Y, n_epochs, optimizer)
test(model, X)
