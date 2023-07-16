from datasets import Dataset, DatasetDict

from data.keys import TRAIN_KEY, VALIDATION_KEY, CONTEXT_KEY

class Numbers():
    def __init__(self, seq_len: int, n_samples: int):
        self.n_samples = n_samples
        self.seq_len = seq_len

        self.train_start = 0
        self.validation_start = n_samples * seq_len

    def build(self) -> DatasetDict:
        return DatasetDict({
            TRAIN_KEY: self.create_train(),
            VALIDATION_KEY: self.create_validation()
        })

    def create_train(self) -> Dataset:
        train_generator = self.generate_dataset_functor(start_index = self.train_start, n_samples = self.n_samples)
        return Dataset.from_generator(train_generator, keep_in_memory=False)

    def create_validation(self) -> Dataset:
        validation_generator = self.generate_dataset_functor(start_index = self.validation_start, n_samples = 100)
        return Dataset.from_generator(validation_generator, keep_in_memory=False)

    def generate_dataset_functor(self, start_index: int, n_samples: int):
        def generate_dataset():
            for start in range(start_index, start_index + self.seq_len * n_samples, self.seq_len):
                yield {CONTEXT_KEY: self.generate_number_sequence(start, self.seq_len)}

        return generate_dataset

    def generate_number_sequence(self, start: int, seq_len: int) -> str:
        """ Generate a sequence of comma seperate numbers whose digits are space seperated """
        return ' , '.join([' '.join(str(i)) for i in range(start, start + seq_len)])
