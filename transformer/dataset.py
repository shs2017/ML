""" Text training, validation, and testing dataset """

import os

from copy import deepcopy
from typing import List

import pytorch_lightning as pl

import torch
from torch import Tensor, LongTensor
from torch.utils.data import Dataset
from torchtext.datasets import WikiText2, WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from config import get_config, DatasetNames

config = get_config()

class TextDataModule(pl.LightningDataModule):
    """Encapsulates a text dataset"""

    def __init__(self, path: str, prepare_data_per_node: bool = True):
        """
        Args:
            dataset (torch.utils.data.Dataset): class of the dataset to use
            prepare_data_per_node (bool): this is used for distributed training.
                                          Refer to the PyTorch Lightning docs.
        """
        super().__init__()
        self.prepare_data_per_node = prepare_data_per_node

        self.batch_size = config.batch_size

        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None  # lazily load vocab

        dataset = self.build_dataset(path)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset
        self.vocab_dataset = deepcopy(self.train_dataset)
        self.unk_token = "<unk>"

    def build_dataset(self, path):
        if config.dataset == DatasetNames.WIKI_TEXT2:
            return WikiText2(path)
        elif config.dataset == DatasetNames.WIKI_TEXT103:
            return WikiText103(path)
        else:
            raise ValueError('Unrecognized dataset')

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
        n_batches = data.size(0) // (self.batch_size * config.max_seq_len)
        data = data[: n_batches * self.batch_size * config.max_seq_len]
        data = data.view(self.batch_size, n_batches, config.max_seq_len)
        data = data.transpose(0, 1)

        return data

    def _get_vocab(self):
        """Compute the vocab for the training set"""
        self.vocab = build_vocab_from_iterator(
            map(self.tokenizer, self.vocab_dataset), specials=[self.unk_token]
        )
        self.vocab.set_default_index(self.vocab[self.unk_token])

    @property
    def vocab_size(self):
        """Number of elements in the vocab"""
        if not self.vocab:
            self._get_vocab()

        return len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        text = [text]
        text = [LongTensor(self.vocab(self.tokenizer(item))) for item in text]
        return text

    def create_test_data(self, text: str) -> Tensor:
        text = self.tokenize(text)
        text = torch.cat(list(filter(lambda t: t.numel(), text)))
        text = text.view(1, 1, len(text))
        text = text.transpose(0, 1)
        return text[0]

    # Standard PyTorch Lightning data module methods implemented
    def prepare_data(self):
        if not self.vocab:  # don't recompute vocab if we already ran `vocab_size`
            self._get_vocab()

    def parse_data(self, data: Tensor):
        return self._batchify(self._data_process(data))

    def transfer_batch_to_device(self, batch, device, _):
        seq_len = min(config.max_seq_len, batch.size(1) - 1)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TextDataModule(path=DATASET_PATH).to(device)
