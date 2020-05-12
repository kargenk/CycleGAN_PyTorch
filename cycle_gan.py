#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import itertools
import matplotlib.pyplot as plt

import networks as net
import dataloader as dl
import utils

# シード値の固定
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ハイパーパラメータの定義
lr = 0.0001
betas = (0.5, 0.999)
batch_size = 1
num_epochs = 200
lambda_cycle = 10.0
lambda_identity = 5.0

# GPUが使用できるか確認
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('使用デバイス：', device)

# 各生成器と判別器をインスタンス化
G_A2B = net.Generator()
G_B2A = net.Generator()
D_A = net.Discriminator()
D_B = net.Discriminator()

# GPUが使用できるならモデルをGPUに載せて初期化
if device == 'cuda:0':
    G_A2B.cuda()
    G_B2A.cuda()
    D_A.cuda()
    D_B.cuda()

print('ネットワークの初期化中...', end='')
G_A2B.apply(utils.weights_init)
G_B2A.apply(utils.weights_init)
D_A.apply(utils.weights_init)
D_B.apply(utils.weights_init)
print('完了！')

# 損失関数を定義
criterion_adversarial = nn.MSELoss()  # 敵対的損失，Mean Squared Error
criterion_cycle = nn.L1Loss()         # サイクル一貫性損失，Mean Absorute Error
criterion_identity = nn.L1Loss()      # 同一性損失

# 最適化手法の設定，Generatorは二つのモデルのパラメータを同時に更新
optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
                         lr=lr, betas=betas)
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=betas)
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=betas)

# 識別器への画像提供(過去の履歴画像:50%，最新の生成画像:50%)のヘルパクラス
fake_A_buffer = utils.PreviousBuffer()
fake_B_buffer = utils.PreviousBuffer()

# データセットのパスを取得
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
