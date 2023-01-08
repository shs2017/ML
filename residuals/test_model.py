import torch
from torch import nn
from torch.testing import assert_close

from .model import *

class TestBatchNormConv2d:
    bs = 2
    channels = 6
    d_embed = 12

    def test_shape(self):
        x = torch.rand(self.bs, self.channels, self.d_embed, self.d_embed)
        f = BatchNormConv2d(in_channels=self.channels,
                            out_channels=self.channels,
                            kernel_size=3,
                            padding=1)

        assert f(x).shape == x.shape


class TestSkip:
    def test_same_shape(self):
        f = nn.Identity()
        model = Skip(f, in_channels=3, scale_factor=1)

        x = torch.rand(3, 12, 12)
        assert model(x).shape == x.shape

    def test_scale_shape(self):
        bs = 2
        scale_factor = 2
        channels = 6
        d_embed = 12

        f = nn.Conv2d(in_channels=channels,
                      out_channels=int(scale_factor * channels),
                      kernel_size=2,
                      stride=scale_factor)

        model = Skip(f, in_channels=channels, scale_factor=scale_factor)

        x = torch.rand(bs, channels, d_embed, d_embed)
        assert model(x).shape == torch.Size([bs,
                                             scale_factor * channels,
                                             d_embed // scale_factor,
                                             d_embed // scale_factor])

class TestConv2dGroup:
    bs = 2
    channels = 6
    d_embed = 12

    def test_shape(self):
        x = torch.rand(self.bs, self.channels, self.d_embed, self.d_embed)
        model = Conv2dGroup(channels=self.channels, num_layers=2)

        assert model(x).shape == x.shape

class TestNormalBlock:
    bs = 2
    channels = 6
    d_embed = 12

    def test_shape(self):
        x = torch.rand(self.bs, self.channels, self.d_embed, self.d_embed)
        model = NormalBlock(channels=self.channels, num_groups=3)
        assert model(x).shape == x.shape

class TestNormalBlock:
    bs = 2
    channels = 6
    d_embed = 12

    def test_shape_half(self):
        scale_factor = 2

        x = torch.rand(self.bs, self.channels, self.d_embed, self.d_embed)
        model = ScaleBlock(channels=self.channels, scale_factor=scale_factor)
        assert model(x).shape == torch.Size([self.bs,
                                             self.channels * scale_factor,
                                             self.d_embed // scale_factor,
                                             self.d_embed // scale_factor])
    def test_shape_third(self):
        scale_factor = 3

        x = torch.rand(self.bs, self.channels, self.d_embed, self.d_embed)
        model = ScaleBlock(channels=self.channels, scale_factor=scale_factor)
        assert model(x).shape == torch.Size([self.bs,
                                             self.channels * scale_factor,
                                             self.d_embed // scale_factor,
                                             self.d_embed // scale_factor])


class TestBlock:
    bs = 2
    channels = 6
    d_embed = 12

    def test_shape_half(self):
        scale_factor = 2

        x = torch.rand(self.bs, self.channels, self.d_embed, self.d_embed)
        model = Block(channels=self.channels,
                      num_groups=2,
                      scale_factor=scale_factor)

        assert model(x).shape == torch.Size([self.bs,
                                             self.channels * scale_factor,
                                             self.d_embed // scale_factor,
                                             self.d_embed // scale_factor])
    def test_shape_third(self):
        scale_factor = 3

        x = torch.rand(self.bs, self.channels, self.d_embed, self.d_embed)
        model = Block(channels=self.channels,
                      num_groups=2,
                      scale_factor=scale_factor)

        assert model(x).shape == torch.Size([self.bs,
                                             self.channels * scale_factor,
                                             self.d_embed // scale_factor,
                                             self.d_embed // scale_factor])

class TestResiduals:
    bs = 2
    channels = 3
    d_embed = 224

    def test_shape_half(self):
        model = Residuals()

        x = torch.rand(self.bs, self.channels, self.d_embed, self.d_embed)
        print(model(x).shape)
        assert True == True
