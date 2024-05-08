import pytorch_lightning as pl

from config import get_config
from model import MainModel
from dataset import get_dataloader

from torchvision.utils import save_image


if __name__ == '__main__':
    config = get_config()

    train_dataloader = get_dataloader(config, train=True)
    validation_dataloader = get_dataloader(config, train=False)

    main_model = MainModel(config)

    trainer = pl.Trainer(max_epochs=config.max_epochs, log_every_n_steps=config.log_every_n_steps)
    trainer.fit(model=main_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    generated_image = main_model.cuda().generate_image(n_samples=16)
    save_image(generated_image, 'image.jpg')