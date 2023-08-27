import torch
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import Dataset

import torchtext
from torchtext.datasets import WikiText2


class TextDataset(Dataset):
    def __init__(self, base_dataset, seq_len: int,
                 device: str, tokenizer_type: str = 'basic_english'):
        self.seq_len = seq_len
        self.stored_seq_len = seq_len + 1

        self.tokenizer_type = tokenizer_type

        dataset, vocab = self.build(base_dataset)
        self.dataset = dataset.to(device)
        self.vocab = vocab

    def __getitem__(self, idx):
        data_in = self.dataset[idx, :-1]
        expected_out = self.dataset[idx, 1:]

        return (data_in, expected_out)

    def __len__(self):
        return len(self.dataset)

    def build(self, dataset):
        batched_dataset = self.parse_dataset(dataset)
        vocab = self.create_vocab(batched_dataset)

        indexed_dataset = Tensor(list(map(vocab, batched_dataset)))
        indexed_dataset = indexed_dataset.to(torch.long)

        return indexed_dataset, vocab

    def parse_dataset(self, dataset):
        tokenized_text = self.tokenize(dataset)
        filtered_text = [x for x in tokenized_text if len(x)]
        flattened_text = [x for y in filtered_text for x in y]

        sequenced_text = []
        for i in range(0, len(flattened_text), self.stored_seq_len):
            sequenced_text.append(flattened_text[i: i + self.stored_seq_len])

        # drop last to guarantee a rectangular array
        return sequenced_text[:-1]

    def tokenize(self, dataset):
        tokenizer = torchtext.data.utils.get_tokenizer(self.tokenizer_type)
        return [tokenizer(x) for x in dataset]

    def create_vocab(self, text):
        return torchtext.vocab.build_vocab_from_iterator(text)


def create_wikitext2_dataset(seq_len: int, device: str):
    train_text, valid_text = WikiText2(root='.', split=('train', 'valid'))

    train_dataset = TextDataset(train_text, seq_len=seq_len, device=device)
    valid_dataset = TextDataset(valid_text, seq_len=seq_len, device=device)

    return (train_dataset, valid_dataset)

# for x in TextDataset(train, seq_len=3):
#     print(x)
#     exit()

# X = Tensor([[0], [1], [2], [3], [2], [1], [0]]).to(torch.long)
# Y = F.one_hot(torch.arange(6, -1, -1), num_classes=10).to(torch.float)
