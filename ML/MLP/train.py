from torch import optim
from torch import nn

import torch.nn.functional as F
import pytorch_lightning as pl

from model import SimpleMLP
from dataset import xor_dataloader

class MainModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.lr = 1e-2
        self.momentum = 0.9
        self.model = SimpleMLP(d_in=2, d_hid=5, d_out=1, n_hid=4, sigma=nn.Sigmoid)

    def training_step(self, batch, _):
        input_data, ground_truth = batch
        prediction = self.model(input_data)
        loss = F.cross_entropy(prediction, ground_truth)        

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(),
                         lr=self.lr,
                         momentum=self.momentum)


main_model = MainModel()

trainer = pl.Trainer(max_epochs=1, log_every_n_steps=10)
trainer.fit(model=main_model, train_dataloaders=xor_dataloader)
