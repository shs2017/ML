""" Trains on transformer-based token prediction model """

import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import Dataset
from model.model import LightningAttentionModel

from cli import arguments
from config import get_config

config = get_config()

class Trainer:
    def __init__(self, vocab_size: int, checkpoint_path: str):
        self.vocab_size = vocab_size
        self.checkpoint_path = checkpoint_path

        if not self.is_checkpoint_valid():
            raise ValueError('Error: Path provided is not valid')

        self.base_trainer = self.create_base_trainer()
        self.model = self.create_model()

    def train(self, dataset: Dataset) -> None:
        self.base_trainer.fit(model=self.model, datamodule=dataset, ckpt_path=self.checkpoint_path)

    def validate(self, dataset: Dataset) -> None:
        self.base_trainer.validate(datamodule=dataset, ckpt_path='best')

    def create_model(self) -> Dataset:
        return LightningAttentionModel(vocab_size=self.vocab_size)

    def create_base_trainer(self) -> pl.Trainer:
        if self.checkpoint_path:
            print(f'Using checkpoint = {self.checkpoint_path}')

        logger = None
        if config.should_log:
            logger = TensorBoardLogger("logs", name="transformer")

        trainer = pl.Trainer(
            accelerator=config.accelerator,
            max_epochs=config.epochs,
            log_every_n_steps=config.log_interval,
            logger=logger,
            callbacks=[],
        )

        return trainer

    def is_checkpoint_valid(self) -> bool:
        return self.checkpoint_path is None or os.path.isfile(self.checkpoint_path)

if __name__ == '__main__':
    from data.dataset import dataset

    print(config)
    args = arguments()
    trainer = Trainer(dataset.vocab_size, args.checkpoint_path)

    trainer.train(dataset)
    trainer.validate(dataset)
