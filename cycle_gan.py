#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

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
input_size = 128  # patch_size = 8(128/2^4)
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

# 真偽判定に用いるテンソルを定義(パッチ形式)
Tensor = torch.cuda.FloatTensor if device == 'cuda:0' else torch.Tensor
target_real = Variable(Tensor(batch_size, 8, 8, 1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batch_size, 8, 8, 1).fill_(0.0), requires_grad=False)

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

# 訓練
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        real_A = batch['A']
        real_B = batch['B']

        ########## Generatorの訓練 ##########
        optimizer_G.zero_grad()

        # identity loss: 同一性損失
        # 本物のBがきたら，G_A2B(B)は本物のBと同じである必要がある
        same_B = G_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * lambda_identity
        # 本物のAがきたら，G_B2A(A)は本物のAと同じである必要がある
        same_A = G_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * lambda_identity

        # adversarial loss: 敵対的損失
        fake_B = G_A2B(real_A)
        pred_fake = D_B(fake_B)
        loss_adversarial_A2B = criterion_adversarial(pred_fake, target_real)

        fake_A = G_B2A(real_B)
        pred_fake = D_A(fake_A)
        loss_adversarial_B2A = criterion_adversarial(pred_fake, target_real)

        # cycle consistency loss: サイクル一貫性損失
        reconstruct_A = G_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(reconstruct_A, real_A) * lambda_cycle

        reconstruct_B = G_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(reconstruct_B, real_B) * lambda_cycle

        # 各損失の合計を算出
        loss_G = loss_identity_A + loss_identity_B +
                 loss_adversarial_A2B + loss_adversarial_B2A +
                 loss_cycle_ABA + loss_cycle_BAB
        
        # 逆伝播して，更新
        loss_G.backward()
        optimizer_G.step()
        ##############################

        ########## Discriminator A の訓練 ##########
        optimizer_D_A.zero_grad()

        # Real loss(本物を本物と判断できれば0)
        pred_real = D_A(real_A)
        loss_D_A_real = criterion_adversarial(pred_real, target_real)

        # Fake loss(偽物を偽物と判断できれば0)
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = D_A(fake_A.detach())
        loss_D_A_fake = criterion_adversarial(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        # 逆伝播して，更新
        loss_D_A.backward()
        optimizer_D_A.step()
        ##############################

        ########## Discriminator B の訓練 ##########
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = D_B(real_B)
        loss_D_B_real = criterion_adversarial(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = D_B(fake_B.detach())
        loss_D_B_fake = criterion_adversarial(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        # 逆伝播して，更新
        loss_D_B.backward()
        optimizer_D_B.step()
        ##############################
