import torch

from config import Config

from torch.optim import SGD
from torch.utils.data import DataLoader

from dataset import create_wikitext2_dataset

from rnn import RNN
from lstm import LSTM
from gru import GRU


if __name__ == '__main__':
    torch.manual_seed(seed=42)

    device = 'cuda'

    batch_size = 20
    n_epochs = 10
    seq_len = 32
    lr = 1e-2

    train_dataset, test_dataset = create_wikitext2_dataset(seq_len=seq_len,
                                                           device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    vocab = train_dataset.vocab
    vocab_size = len(train_dataset.vocab)

    config = Config(
        d_in=256,
        d_hid=256,
        d_out=vocab_size,
        n_layers=5,
        vocab_size=vocab_size,
        device=device
    )

    def optimizer_fn(lr: float):
        return SGD

    rnn = RNN(config, optimizer_fn=optimizer_fn(lr))
    lstm = LSTM(config, optimizer_fn=optimizer_fn(lr))
    gru = GRU(config, optimizer_fn=optimizer_fn(lr))

    models = [lstm]

    for model in models:
        print(f'*** Training {model.name} ***')
        model.train(dataset=train_dataloader, n_epochs=n_epochs, vocab=vocab)

        print(f'*** Testing {model.name} *** ')
        model.test(dataset=test_dataloader, n_layers=n_layers, vocab=vocab)
