#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networks as net
import torch

# Generatorの動作チェック
G = net.Generator(3)
input_image = torch.randn(128, 128, 3)
input_image = input_image.view(1, 3, 128, 128)
fake_image = G(input_image)
print(fake_image.shape)
