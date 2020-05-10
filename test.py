#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import networks as net

# Generatorの動作チェック
G = net.Generator(3)
input_image = torch.randn(128, 128, 3)
input_image = input_image.view(1, 3, 128, 128)
fake_image = G(input_image)
print('fake image:', fake_image.shape)

# Discriminatorの動作チェック
D = net.Discriminator(3)
d_out = D(fake_image)
print(nn.Sigmoid()(d_out).shape)  # 出力にSigmoidをかけて[0, 1]に変換
