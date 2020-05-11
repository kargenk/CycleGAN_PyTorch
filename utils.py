#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import random

def weights_init(m):
    """
    ネットワークの初期化を行う関数．
    
    Parameters
    ----------
    m : torch.nn.Model
        ネットワークモデル
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    # elif classname.find('InstanceNorm') != -1:
    #     # InstanceNorm2dの初期化
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)

class PreviousBuffer():
    """
    過去に生成された画像を格納するバッファ．
    最新の生成器で生成された画像ではなく，過去に生成された画像の履歴を使用して識別器を更新することができる．
    (生成器が先に学習を終えてしまった場合に対応するため？)

    Attributes
    ----------
    max_size : int
        画像の保存履歴の最大数
    num_imgs : int
        保存する画像数を管理する数
    images : torch.Tensor
        画像
    """

    def __init__(self, max_size=50):
        self.max_size = max_size
        if self.max_size > 0:
            self.num_imgs = 0
            self.images = []

    def push_and_pop(self, images):
        """
        識別器の訓練に用いる画像を返す関数．
        50%の確率で過去の生成画像を返して，今の生成画像をバッファに加える．
        残りの50%の確率で今の生成画像を返す．

        Parameters
        ----------
        images : torch.Tensor
            最新の識別器から生成された画像
        
        Returns
        ----------
        return_images : torch.Tensor
            過去の生成画像の履歴画像(50%)，または今回の生成画像(50%)
        """

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.max_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                # 履歴から返す(50%)か，今のものを返す(50%)か
                if random.uniform(0, 1) > 0.5:
                    random_id = random.randint(0, self.max_size - 1)
                    temp = self.images[random_id].clone()
                    self.images[random_id] = image  # 今の生成画像を格納
                    return_images.append(temp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, dim=0)

        return return_images
