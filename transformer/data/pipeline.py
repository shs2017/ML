from dataclasses import dataclass
from itertools import chain

from data.base_dataset import BaseDataset, CONTEXT_KEY
from data.tokenizer import Tokenizer

from config import get_config

config = get_config()

@dataclass
class PipelineOptions:
    backing_format: str
    max_seq_len: int

class Pipeline:
    FORMATTED_ID_KEY = 'formatted_ids'
    INPUT_ID_KEY = 'input_ids'

    def __init__(self, dataset: BaseDataset, tokenizer: Tokenizer, options: PipelineOptions):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.options = options

    def create_updated_dataset(self):
        return {key: self.format_dataset(value) for key, value in self.dataset.items()}

    def format_dataset(self, dataset):
        if dataset is None:
            return None

        dataset = dataset.map(self.tokenizer.tokenize, batched=True, remove_columns=[CONTEXT_KEY], num_proc=config.num_workers)
        dataset = dataset.map(self.group, batched=True, num_proc=config.num_workers)
        dataset = dataset.with_format(self.options.backing_format)
        return dataset[self.FORMATTED_ID_KEY]

    def group(self, data):
        groups = {col_key: self.group_column(col_data) for col_key, col_data in data.items()}
        groups[self.FORMATTED_ID_KEY] = groups[self.INPUT_ID_KEY].copy()
        return groups

    def group_column(self, data):
        flattened_data = self.flatten_column(data)
        return self.split_into_blocks(flattened_data)

    def flatten_column(self, data):
        return list(chain.from_iterable(data))

    def split_into_blocks(self, data):
        block_size = self.options.max_seq_len
        total_len = self.calculate_truncated_len(data)
        return [data[i : i+block_size] for i in range(0, total_len, block_size)]

    def calculate_truncated_len(self, data):
        block_size = self.options.max_seq_len
        return len(data) // block_size * block_size
