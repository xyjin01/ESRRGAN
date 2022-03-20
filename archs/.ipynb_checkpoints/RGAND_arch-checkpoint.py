import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

import sys
import datetime
import time
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class RGAN128(nn.Module):
    """Realness GAN discriminator with input size 128 x 128.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
        num_outcomes(int): number of outcomes, stands for the discrete distribution of discriminator output,
                           will be used for the RGAN loss computing.
            Default：20
    """

    def __init__(self, num_in_ch, num_feat, num_outcomes):
        super(RGAN128, self).__init__()
        self.num_outcomes = num_outcomes
        self.num_in_ch = num_in_ch
        self.num_feat = num_feat
        norm = spectral_norm
        self.conv0_0 = norm(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True))
        self.conv0_1 = norm(nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False))

        self.conv1_0 = norm(nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False))
        self.conv1_1 = norm(nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False))

        self.conv2_0 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False))
        self.conv2_1 = norm(nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False))

        self.conv3_0 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False))
        self.conv3_1 = norm(nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False))

        self.conv4_0 = norm(nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False))
        self.conv4_1 = norm(nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False))

        self.linear1 = norm(nn.Linear(num_feat * 8 * 4 * 4, num_outcomes, bias=False))

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 128 and x.size(3) == 128, (f'Input spatial size must be 128x128, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.conv0_1(feat))  # output spatial size: (64, 64)

        feat = self.lrelu(self.conv1_0(feat))
        feat = self.lrelu(self.conv1_1(feat))  # output spatial size: (32, 32)

        feat = self.lrelu(self.conv2_0(feat))
        feat = self.lrelu(self.conv2_1(feat))  # output spatial size: (16, 16)

        feat = self.lrelu(self.conv3_0(feat))
        feat = self.lrelu(self.conv3_1(feat))  # output spatial size: (8, 8)

        feat = self.lrelu(self.conv4_0(feat))
        feat = self.lrelu(self.conv4_1(feat))  # output spatial size: (4, 4)

        feat = feat.view(-1, self.num_feat * 8 * 4 * 4)
        out = self.linear1(feat).view(-1, self.num_outcomes)

        return out
