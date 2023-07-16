""" Text training, validation, and testing dataset """

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader

from data.base_dataset import BaseDataset, TRAIN_KEY, VALIDATION_KEY, TEST_KEY
from data.tokenizer import Tokenizer
from data.pipeline import Pipeline, PipelineOptions

from config import get_config, DatasetNames

config = get_config()

class Dataset(pl.LightningDataModule):
    """Encapsulates a text dataset"""

    def __init__(self, prepare_data_per_node: bool = True):
        super().__init__()

        self.prepare_data_per_node = prepare_data_per_node
        self.batch_size = config.batch_size

        self.tokenizer = self.create_tokenizer()

        self.name = config.dataset.value.name
        self.sub_name = config.dataset.value.sub_name

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def create_tokenizer(self):
        return Tokenizer(name='bert-base-uncased')

    def create_dataset(self, tokenizer):
        base_dataset = BaseDataset(name=self.name, sub_name=self.sub_name)

        options = PipelineOptions(backing_format='torch', max_seq_len=config.max_seq_len)
        pipeline = Pipeline(dataset=base_dataset, tokenizer=tokenizer, options=options)
        return pipeline.create_updated_dataset()

    def transfer_to(self, device: str):
        self.device = device
        return self

    # Standard PyTorch Lightning data module methods implemented
    def setup(self, stage):
        dataset = self.create_dataset(self.tokenizer)
        self.train_dataset = dataset[TRAIN_KEY]
        self.validation_dataset = dataset[VALIDATION_KEY]
        self.test_dataset = dataset[TEST_KEY]

    def transfer_batch_to_device(self, batch, device, _):
        batch = batch.to(device)
        seq_len = min(config.max_seq_len, batch.size(1) - 1)
        data = batch[:, 0 : seq_len]
        target = batch[:, 1 : 1 + seq_len]
        return data, target

    def create_dataloader(self, dataset, shuffle: bool):
        return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=shuffle)

    def train_dataloader(self):
        return self.create_dataloader(self.train_dataset, shuffle = config.shuffle)

    def val_dataloader(self):
        return self.create_dataloader(self.validation_dataset, shuffle = False)

    def test_dataloader(self):
        if self.test_dataset is None:
            print('No test dataset specified, using the validation dataset instead')
            return self.val_dataloader()

        return self.create_dataloader(self.test_dataset, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Dataset().transfer_to(device)
