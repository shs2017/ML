import pytorch_lightning as pl

from config import get_config
from dataset import SegmentationDataset, compute_class_weights
from model import MainModel


if __name__ == '__main__':
    config = get_config()
    dataset = SegmentationDataset(config)

    # can use compute_class_weights here as well for a class weighted loss
    main_model = MainModel(config, class_weights=None)

    train_dataset = dataset.retrieve_train_data()
    val_dataset = dataset.retrieve_val_data()

    trainer = pl.Trainer(max_epochs=config.max_epochs, log_every_n_steps=config.log_every_n_steps)
    trainer.fit(model=main_model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)
