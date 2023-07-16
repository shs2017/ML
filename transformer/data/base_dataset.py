from datasets import Dataset, DatasetDict, load_dataset
from data.numbers_dataset import Numbers
from data.keys import TRAIN_KEY, VALIDATION_KEY, TEST_KEY, CONTEXT_KEY, INPUT_ID_KEY

class BaseDataset:
    def __init__(self, name: str, sub_name: str = None):
        if name == 'numbers':
            self.dataset = Numbers(seq_len=512, n_samples=1_000).build()
        else:
            self.dataset = load_dataset(name, sub_name, keep_in_memory=False)

    def items(self):
        return {
            TRAIN_KEY: self.train,
            VALIDATION_KEY: self.validation,
            TEST_KEY: self.test
        }.items()

    @property
    def train(self):
        return self.retrieve_data_by_key(key=TRAIN_KEY, default=None)

    @property
    def validation(self):
        return self.retrieve_data_by_key(key=VALIDATION_KEY, default=None)

    @property
    def test(self):
        return self.retrieve_data_by_key(key=TEST_KEY, default=None)

    def retrieve_data_by_key(self, key, default):
        return self.dataset[key] if key in self.dataset else default
