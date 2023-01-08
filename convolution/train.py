import pytorch_lightning as pl

import torch.nn.functional as F

from torch import nn
from torch import optim

from model import CNNNet
from dataset import train_dataloader

class MainModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = CNNNet()
        self.lr = 1e-2
        self.momentum = 0.9

    def training_step(self, batch, _):
        img, target = batch

        prediction = self.model(img)
        loss = F.cross_entropy(prediction, target)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(),
                         lr=self.lr,
                         momentum=self.momentum)


main_model = MainModel()
print(main_model)

trainer = pl.Trainer(max_epochs=25, log_every_n_steps=100)
trainer.fit(model=main_model, train_dataloaders=train_dataloader)
