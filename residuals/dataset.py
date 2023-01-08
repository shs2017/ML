import os

from torch.utils.data import DataLoader
from torchvision.datasets import Caltech256
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

num_workers=1
batch_size=8


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.dirname(BASE_PATH)
DATASET_PATH = os.path.join(ROOT_PATH, 'datasets')

class ConvertImage:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, img):
        return img.convert(self.mode)

preprocess = Compose([
    Resize((224, 224)),
    ConvertImage('RGB'),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = Caltech256(root=DATASET_PATH, transform=preprocess)
test_dataset = Caltech256(root=DATASET_PATH, transform=preprocess)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
