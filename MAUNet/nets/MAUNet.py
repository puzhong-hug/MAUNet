# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 10:59 上午
# @File    : MAU-Net.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F
from nets.DCT13 import DCT13
from nets.DCT24 import DCT24


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    # Get input sizes
    _, c, h, w = x.size()
    # Create per-element exponential value sum : Tensor [b x 1 x h x w]
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x h x w] -> [b x c x h' x w']
    return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))

class T_CCA(nn.Module):
    """
    T-CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        soft_pool_x = soft_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(soft_pool_x)
        soft_pool_g = soft_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(soft_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = T_CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h


class MAUNet(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.attention1 = CoordAttention(in_channels,in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.attention2 = CoordAttention(in_channels*2, in_channels*2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.attention3 = CoordAttention(in_channels*4, in_channels*4)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.attention4 = CoordAttention(in_channels*8, in_channels*8)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.mtc = DCT13(config, vis, img_size,
                                      channel_num=[in_channels, in_channels*4],
                                      patchSize=config.patch_sizes)
        self.mtc1 = DCT24(config, vis, img_size,
                                     channel_num=[in_channels*2, in_channels*8],
                                     patchSize=config.patch_sizes1)
        self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss

    def forward(self, x):
        x = x.float()
        d1 = self.inc(x)
        d1 = self.attention1(d1)
        d2 = self.down1(d1)
        # d2 = self.attention2(d2)
        d3 = self.down2(d2)
        # d3 = self.attention3(d3)
        d4 = self.down3(d3)
        # d4 = self.attention4(d4)
        d5 = self.down4(d4)
        o1,o3,att_weights = self.mtc(d1,d3)
        o2,o4,att_weights = self.mtc1(d2,d4)
        u4 = self.up4(d5, o4)
        u3 = self.up3(u4, o3)
        u2 = self.up2(u3, o2)
        u1 = self.up1(u2, o1)
        if self.n_classes ==1:
            logits = self.last_activation(self.outc(u1))
        else:
            logits = self.outc(u1) # if nusing BCEWithLogitsLoss or class>1
        if self.vis: # visualize the attention maps
            return logits, att_weights
        else:
            return logits


