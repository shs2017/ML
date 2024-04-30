from dataclasses import dataclass

@dataclass
class Config():
    # Training Parameters
    lr: float
    momentum: float
    max_epochs: int
    log_every_n_steps: int
    batch_size: int
    dataset_folder_name: str

    # Model Parameters
    in_channels: int
    out_channels: int
    kernel_size: int
    downscale_kernel_size: int
    upscale_kernel_size: int


def get_config():
    return Config(
        lr=1e-3,
        momentum=0.9,
        max_epochs=200,
        log_every_n_steps=100,
        batch_size=16,
        dataset_folder_name='datasets',

        in_channels=1,
        out_channels=24,
        kernel_size=3,
        downscale_kernel_size=2,
        upscale_kernel_size=2
    )
