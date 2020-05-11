#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

import networks as net
import dataloader as dl

##### Generatorの動作チェック #####
G = net.Generator(3)
input_image = torch.randn(128, 128, 3)
input_image = input_image.view(1, 3, 128, 128)
fake_image = G(input_image)
print('fake image:', fake_image.shape)

##### Discriminatorの動作チェック #####
D = net.Discriminator(3)
d_out = D(fake_image)
print('Discriminator output', nn.Sigmoid()(d_out).shape)  # 出力にSigmoidをかけて[0, 1]に変換

##### Dataset, DataLoaderの動作チェック #####
train_img_A, train_img_B = dl.make_datapath_list(is_train=True)

# Datasetを作成
mean = (0.5,)
std = (0.5,)
train_dataset = dl.UnpairedDataset(train_img_A, train_img_B,
                                   transform=dl.ImageTransform(mean, std))

# DataLoaderを作成
batch_size = 1
train_dataloader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

# 動作の確認
batch_iterator = iter(train_dataloader)  # イテレータに変換
images = next(batch_iterator)            # 1番目の要素を取り出す
# print(images)
plt.subplot(121)
dl.imshow(images['A'])
plt.subplot(122)
dl.imshow(images['B'])
plt.show()
