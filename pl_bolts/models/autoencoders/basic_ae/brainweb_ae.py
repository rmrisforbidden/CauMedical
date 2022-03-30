import torch
from torch import nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))


def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))


class EncoderBlock(nn.Module):
    """ResNet block, copied from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        return out


class DecoderBlock(nn.Module):
    """ResNet block, but convs replaced with resize convs, and channel increase is in second conv, not first."""

    def __init__(self, inplanes, planes, latent_dim, input_height, scale=1, upsample=None):
        super().__init__()
        self.upscale_factor = 4
        self.conv1 = resize_conv3x3(inplanes, inplanes, scale)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = resize_conv3x3(inplanes, planes, scale)
        self.bn3 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.linear = nn.Linear(latent_dim, inplanes * 4 * 4)
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)
        self.conv1 = nn.Conv2d(254, 254, kernel_size=3, stride=1, padding=1, bias=False)
        self.upscale = Interpolate(scale_factor=1)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), 254, 4, 4)
        out = self.upscale1(out)

        out = self.conv1(out)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        out = self.upscale(out)
        out = self.conv1(out)

        return out


def simple_encoder(first_conv, maxpool1):
    return EncoderBlock(inplanes=254, planes=254, stride=2, downsample=None)


def simple_decoder(latent_dim, input_height, first_conv, maxpool1):
    return DecoderBlock(inplanes=254, planes=254, latent_dim=latent_dim, input_height=input_height, scale=2)
