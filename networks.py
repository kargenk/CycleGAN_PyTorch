#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

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
    """ UNet構造 """

    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        self.conv1 = conv2d(in_channels, 32)    # (N, 3, 128, 128) -> (N, 32, 64, 64)
        self.conv2 = conv2d(32, 64)             # (N, 32, 64, 64) -> (N, 64, 32, 32)
        self.conv3 = conv2d(64, 128)            # (N, 64, 32, 32) -> (N, 128, 16, 16)
        self.conv4 = conv2d(128, 256)           # (N, 128, 16, 16) -> (N, 256, 8, 8)
        
        # ここから残差接続によりoutchannelsが2倍
        self.deconv1 = deconv2d_res(256, 128)   # (N, 256, 8, 8) -> (N, 128+128, 16, 16), skip
        self.deconv2 = deconv2d_res(128*2, 64)  # (N, 256, 16, 16) -> (N, 64+64, 32, 32), skip
        self.deconv3 = deconv2d_res(64*2, 32)   # (N, 128, 32, 32) -> (N, 32+32, 64, 64), skip
        self.last = nn.Sequential(
            nn.ConvTranspose2d(32*2, 64, kernel_size=2, stride=2),            # (N, 64, 64, 64) -> (N, 64, 128, 128)
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),  # (N, 64, 128, 128) -> (N, 3, 128, 128)
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

class Discriminator(nn.Module):
    """
    PatchGANの構造．
    CNNから得られる特徴マップの1pixelは，入力画像のある領域(Receptive field)の影響を受けた値．
    したがって，「最終出力をあるサイズ(入力画像の1/2^4)の特徴マップにして，各pixelで真偽判定をする」ことと，
    「入力画像をpatchにして，各patchの出力で真偽判定をすること」は等価．
    """

    def __init__(self, in_channels=3, out_channels=1):
        super(Discriminator, self).__init__()
        
        self.conv1 = conv2d(in_channels, 64)  # (N, 3, 128, 128) -> (N, 64, 64, 64)
        self.conv2 = conv2d(64, 128)          # (N, 64, 64, 64) -> (N, 128, 32, 32)
        self.conv3 = conv2d(128, 256)         # (N, 128, 32, 32) -> (N, 256, 16, 16)
        self.conv4 = conv2d(256, 512)         # (N, 256, 16, 16) -> (N, 512, 8, 8)

        # 真贋判定はPatch形式
        self.validity = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)  # (N, 512, 8, 8) -> (N, 1, 8, 8)
    
    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        out = self.validity(d4)

        return out
