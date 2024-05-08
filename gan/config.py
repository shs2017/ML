from dataclasses import dataclass


GRAYSCALE_DIM = 1
RGB_DIM = 3


@dataclass
class Config():
    # Training Parameters
    lr: float
    betas: tuple[float, float]

    max_epochs: int
    log_every_n_steps: int
    batch_size: int
    dataset_folder_name: str

    # Model Parameters
    base_channels: int
    hidden_channels: int
    image_input_channels: int
    discriminator_output_channels: int

    kernel_size: int
    downscale_kernel_size: int
    upscale_kernel_size: int

    weight_clip_value: float


def get_config():
    return Config(
        lr=2e-4,
        betas=(0.5, 0.999),

        max_epochs=25,
        log_every_n_steps=250,
        batch_size=256,
        dataset_folder_name='datasets',

        base_channels=64,
        hidden_channels=128,
        image_input_channels=RGB_DIM,
        discriminator_output_channels=1,

        kernel_size=4,
        downscale_kernel_size=2,
        upscale_kernel_size=2,

        weight_clip_value=0.01
    )
