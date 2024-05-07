import pytorch_lightning as pl

import torch

import torch.nn.functional as F
from torch import nn, optim, Tensor

from torchvision.utils import save_image

from config import Config

from math import exp

class MainModel(pl.LightningModule):
    def __init__(self, config: Config, class_weights: torch.Tensor = None):
        super().__init__()
        self.automatic_optimization = False # allows us to control how gan is trained

        self.lr = config.lr
        self.betas = config.betas
        self.hidden_channels = config.hidden_channels
        self.weight_clip_value = config.weight_clip_value

        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def random_hidden_vector(self, batch_size: int) -> Tensor:
        return torch.randn(size=(batch_size, self.hidden_channels, 1, 1)).cuda()

    def compute_discriminator_loss(self, batch):
        real_image, class_label = batch

        batch_size = real_image.size(0)

        real_classification = self.discriminator(real_image)

        generated_image = self.generator(self.random_hidden_vector(batch_size))
        generated_classification = self.discriminator(generated_image)

        loss = real_classification.mean() - generated_classification.mean()
        return loss

    def compute_generator_loss(self, batch):
        image, class_label = batch

        batch_size = image.size(0)

        generated_image = self.generator(self.random_hidden_vector(batch_size))
        generated_classification = self.discriminator(generated_image)

        loss = generated_classification.mean()
        return loss

    def clip_discriminator_parameters(self):
        for parameter in self.discriminator.parameters():
            parameter.data.clamp_(-self.weight_clip_value, self.weight_clip_value)

    def training_step(self, batch, _):
        generator_optimizer, discriminator_optimizer = self.optimizers()

        # Train Discriminator
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()

        discriminator_loss = self.compute_discriminator_loss(batch)
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()
        self.clip_discriminator_parameters()

        # Train Generator
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()
        generator_loss = self.compute_generator_loss(batch)
        self.manual_backward(generator_loss)
        generator_optimizer.step()

        self.log('train_real_loss', discriminator_loss, prog_bar=True)
        self.log('train_generated_loss', generator_loss, prog_bar=True)


    def configure_optimizers(self):
        generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        return generator_optimizer, discriminator_optimizer

    def generate_image(self):
        generated_image = self.generator(self.random_hidden_vector(64))
        save_image(generated_image, 'image.jpg')


class ConvTranposeGroup(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.sigma = nn.LeakyReLU()

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.sigma(x)
        return x


class ConvGroup(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.sigma = nn.LeakyReLU()

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.sigma(x)
        return x


class Generator(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        in_channels = config.hidden_channels
        base_channels = config.base_channels
        out_channels = config.image_input_channels

        self.f = nn.Sequential(*[
            ConvTranposeGroup(in_channels, 8 * base_channels, config.kernel_size, 1, 0),
            ConvTranposeGroup(8 * base_channels, 4 * base_channels, config.kernel_size, config.upscale_kernel_size, 1),
            ConvTranposeGroup(4 * base_channels, 2 * base_channels, config.kernel_size, config.upscale_kernel_size, 1),
            ConvTranposeGroup(2 * base_channels, base_channels, config.kernel_size, config.upscale_kernel_size, 1),
            nn.ConvTranspose2d(base_channels, out_channels, config.kernel_size, config.upscale_kernel_size, 1),
            nn.Tanh()
        ])

    def forward(self, x: Tensor) -> Tensor:
        return self.f(x)


class Discriminator(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        in_channels = config.image_input_channels
        base_channels = config.base_channels
        out_channels = config.discriminator_output_channels

        self.f = nn.Sequential(*[
            ConvGroup(in_channels, base_channels, config.kernel_size, config.downscale_kernel_size, 1),
            ConvGroup(base_channels, 2 * base_channels, config.kernel_size, config.downscale_kernel_size, 1),
            ConvGroup(2 * base_channels, 4 * base_channels, config.kernel_size, config.downscale_kernel_size, 1),
            ConvGroup(4 * base_channels, 8 * base_channels, config.kernel_size, config.downscale_kernel_size, 1),
            nn.Conv2d(8 * base_channels, out_channels, config.kernel_size, 1, 0, bias=False)
        ])

    def forward(self, x: Tensor) -> Tensor:
        return self.f(x)


if __name__ == '__main__':
    # Sanity check if running this script directly
    from config import get_config

    config = get_config()

    r = torch.rand(256, 64, 1, 1)
    generator = Generator(config)
    generator_out = generator(r)
    print(generator_out.shape)

    r = torch.rand(256, 1, 32, 32)
    discriminator = Discriminator(config)
    discriminator_out = discriminator(r)
    print(discriminator_out.shape)

    print(generator)
    print(discriminator)