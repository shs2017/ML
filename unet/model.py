import torch
import torch.nn.functional as F

from torch import nn, Tensor

from config import Config

class UNet(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        channels1 = config.in_channels
        channels2 = 64
        channels3 = 128
        channels4 = 256
        channels5 = 512
        channels6 = 1024

        self.down1 = ConvDown(channels1, channels2, config.kernel_size, config.downscale_kernel_size)
        self.down2 = ConvDown(channels2, channels3, config.kernel_size, config.downscale_kernel_size)
        self.down3 = ConvDown(channels3, channels4, config.kernel_size, config.downscale_kernel_size)
        self.down4 = ConvDown(channels4, channels5, config.kernel_size, config.downscale_kernel_size)

        self.conv1 = nn.Conv2d(
            in_channels=channels5,
            out_channels=channels6,
            kernel_size=config.kernel_size
        )

        self.conv2 = nn.Conv2d(
            in_channels=channels6,
            out_channels=channels6,
            kernel_size=config.kernel_size
        )

        self.up1 = ConvUp(channels6, channels5, config.kernel_size, config.upscale_kernel_size)
        self.up2 = ConvUp(channels5, channels4, config.kernel_size, config.upscale_kernel_size)
        self.up3 = ConvUp(channels4, channels3, config.kernel_size, config.upscale_kernel_size)
        self.up4 = ConvUp(channels3, channels2, config.kernel_size, config.upscale_kernel_size)

        self.out_conv = nn.Conv2d(
            in_channels=channels2,
            out_channels=config.out_channels,
            kernel_size=1
        )

    def crop(self, x: Tensor, size: int):
        return x[:, :, size:-size, size:-size]

    def forward(self, x: Tensor) -> Tensor:
        downscaled_out1, copied_out1 = self.down1(x)
        downscaled_out2, copied_out2 = self.down2(downscaled_out1)
        downscaled_out3, copied_out3 = self.down3(downscaled_out2)
        downscaled_out4, copied_out4 = self.down4(downscaled_out3)

        copied_out1 = self.crop(copied_out1, size=88)
        copied_out2 = self.crop(copied_out2, size=40)
        copied_out3 = self.crop(copied_out3, size=16)
        copied_out4 = self.crop(copied_out4, size=4)

        x = self.conv1(downscaled_out4)
        x = self.conv2(x)

        x = self.up1(copied_out4, x)
        x = self.up2(copied_out3, x)
        x = self.up3(copied_out2, x)
        x = self.up4(copied_out1, x)

        x = self.out_conv(x)

        return x

class ConvDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, downscale_kernel_size: int) -> None:
        super().__init__()

        self.conv = ConvGroup(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=downscale_kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        copied_out = self.conv(x)
        downscaled_out = self.pool(copied_out)
        return downscaled_out, copied_out

class ConvUp(nn.Module):
    CHANNEL_AXIS = 1

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, upscale_kernel_size: int) -> None:
        super().__init__()

        self.upscale = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upscale_kernel_size,
            stride=2
        )

        self.conv = ConvGroup(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=kernel_size)


    def cat_features(self, x: Tensor, y: Tensor):
        return torch.cat((x, y), dim=self.CHANNEL_AXIS)

    def forward(self, copied: Tensor, x: Tensor) -> Tensor:
        x = self.upscale(x)
        x = self.cat_features(copied, x)
        x = self.conv(x)
        return x

class ConvGroup(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )


    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


if __name__ == '__main__':
    from config import get_config

    config = get_config()

    r = torch.rand(10, 1, 572, 572)
    conv = UNet(config)
    out = conv(r)

    print(f'{out.shape=}')
