#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class conv2d(nn.Module):
    """
    サイズを1/2にするダウンサンプリング層．
    バッチ全体ではなく，各チャンネル毎に正規化するInstance Normalizationを用いた畳み込みをしている．
    Instance Normalizationは画風変換や，画像から画像への翻訳に良い結果をもたらす．
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(conv2d, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class deconv2d_res(nn.Module):
    """ サイズを二倍にする転置畳み込み(アップサンプリング)層． """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(deconv2d_res, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.layer(x1)
        out = torch.cat([x2, x1], dim=1)  # スキップ接続
        return out

class Generator(nn.Module):
    
    def __init__(self, in_channels, out_channels=3):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv2d(in_channels, 32)
        self.conv2 = conv2d(32, 64)
        self.conv3 = conv2d(64, 128)
        self.conv4 = conv2d(128, 256)
        self.deconv1 = deconv2d_res(256, 128)  # ここから残差接続によりoutchannelsが2倍
        self.deconv2 = deconv2d_res(128*2, 64)
        self.deconv3 = deconv2d_res(64*2, 32)
        self.last = nn.Sequential(
            nn.ConvTranspose2d(32*2, 64, kernel_size=2, stride=2),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        u1 = self.deconv1(d4, d3)
        u2 = self.deconv2(u1, d2)
        u3 = self.deconv3(u2, d1)
        out = self.last(u3)

        return out

