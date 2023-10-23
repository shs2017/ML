from dataclasses import dataclass

@dataclass
class Config():
    # Training Parameters
    lr: float
    momentum: float
    max_epochs: int
    log_every_n_steps: int

    # Model Parameters
    in_channels: int
    out_channels: int
    kernel_size: int
    downscale_kernel_size: int
    upscale_kernel_size: int


def get_config():
    return Config(
        lr=1e-3,
        momentum=0.95,
        max_epochs=25,
        log_every_n_steps=100,
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        downscale_kernel_size=2,
        upscale_kernel_size=2

    )
