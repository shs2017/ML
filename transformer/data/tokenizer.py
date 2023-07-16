from torch import LongTensor
from transformers import AutoTokenizer

from data.base_dataset import CONTEXT_KEY, INPUT_ID_KEY

class Tokenizer:
    def __init__(self, name):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)

    def tokenize(self, data):
        return self.tokenizer(data[CONTEXT_KEY])

    def untokenize(self, data):
        out = self.tokenizer.convert_ids_to_tokens(data)
        out = self.tokenizer.convert_tokens_to_string(out)
        return out

    def convert_words_to_ids(self, data):
        return LongTensor([self.tokenizer(data)[INPUT_ID_KEY]])

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
