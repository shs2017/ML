import torch

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

num_workers = 8
torch.set_default_dtype(torch.float32)

class XORDataset(Dataset):
    def __init__(self):
        self.data = Tensor([[0, 0],
                            [0, 0],
                            [1, 1],
                            [1, 1]])

        self.ground_truth = Tensor([0, 1, 1, 0]).unsqueeze(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index], self.ground_truth[index]

xor_dataloader = DataLoader(XORDataset(), num_workers=num_workers)
