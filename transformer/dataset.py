""" Text training, validation, and testing dataset """

import os

from copy import deepcopy

import pytorch_lightning as pl

import torch
from torch import Tensor, LongTensor
from torch.utils.data import Dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TextDataModule(pl.LightningDataModule):
    """Encapsulates a text dataset"""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        dataset: Dataset,
        prepare_data_per_node: bool = True,
    ):
        """
        Args:
            batch_size (int): batch size
            seq_len (int): maximum sequence length
            dataset (torch.utils.data.Dataset): class of the dataset to use
            prepare_data_per_node (bool): refer to pytorch lightning docs (useful for distributed
                                        training when set to True)
        """
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.prepare_data_per_node = prepare_data_per_node  # for distributed training
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None  # lazily load vocab
        self.train_dataset, self.val_dataset, self.test_dataset = dataset
        self.vocab_dataset = deepcopy(self.train_dataset)
        self.unk_token = "<unk>"

    def to(self, device: str):
        """
        Args:
            device (str): pytorch string representation of device to preload data to.
                          Setting this to 'cuda', for example, will load the entire dataset
                          into memory
        """
        self.device = device
        return self

    def _data_process(self, text_iter):
        """Converts raw text into a flat Tensor."""
        data = [LongTensor(self.vocab(self.tokenizer(item))) for item in text_iter]
        return torch.cat(list(filter(lambda t: t.numel(), data)))

    def _batchify(self, data: Tensor) -> Tensor:
        """Prepares dataset for training batches"""
        n_batches = data.size(0) // (self.batch_size * self.seq_len)
        data = data[: n_batches * self.batch_size * self.seq_len]
        data = data.view(self.batch_size, n_batches, self.seq_len)
        data = data.transpose(0, 1)

        return data

    def _get_vocab(self):
        """Compute the vocab for the training set"""
        self.vocab = build_vocab_from_iterator(
            map(self.tokenizer, self.vocab_dataset), specials=[self.unk_token]
        )
        self.vocab.set_default_index(self.vocab[self.unk_token])

    @property
    def ignore_index(self):
        """Vocab index used for the "<unk>" token"""
        if not self.vocab:
            self._get_vocab()

        return self.vocab[self.unk_token]

    @property
    def vocab_size(self):
        """Number of elements in the vocab"""
        if not self.vocab:
            self._get_vocab()

        return len(self.vocab)

    # Standard PyTorch Lightning data module methods implemented

    def prepare_data(self):
        if not self.vocab:  # don't recompute vocab if we already ran `vocab_size`
            self._get_vocab()

    def parse_data(self, data: Tensor):
        return self._batchify(self._data_process(data))

    def transfer_batch_to_device(self, batch, device, _):
        seq_len = min(self.seq_len, batch.size(1) - 1)
        data = batch[:, 0:seq_len]
        target = batch[:, 1 : 1 + seq_len].reshape(-1)
        return data, target

    def setup(self, stage: str):
        self.train_data = self.parse_data(self.train_dataset)
        self.val_data = self.parse_data(self.val_dataset)
        self.test_data = self.parse_data(self.test_dataset)

    def train_dataloader(self):
        return self.train_data.to(self.device)

    def val_dataloader(self):
        return self.val_data.to(self.device)

    def test_dataloader(self):
        return self.test_data.to(self.device)


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.dirname(BASE_PATH)
DATASET_PATH = os.path.join(ROOT_PATH, "datasets")

BATCH_SIZE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TextDataModule(
    batch_size=BATCH_SIZE, seq_len=35, dataset=WikiText2(DATASET_PATH)
).to(device)
