#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

def make_datapath_list(is_train):
    """
    学習用の画像データセットへのファイルパスのリストを作成する．

    Parameters
    ----------
    is_train : bool
        訓練画像か否かのフラグ
    
    Returns
    ----------
    img_list : list
        画像データセットへのファイルパスのリスト
    """

    img_list_A = list()
    img_list_B = list()

    root_dir = os.path.join('dataset', 'apple2orange')

    if is_train:
        dir_A = os.path.join(root_dir, 'trainA')
        dir_B = os.path.join(root_dir, 'trainB')
    else:
        dir_A = os.path.join(root_dir, 'testA')
        dir_B = os.path.join(root_dir, 'testB')

    for fname in os.listdir(dir_A):
        if fname.endswith('.jpg'):
            path = os.path.join(dir_A, fname)
            img_list_A.append(path)
    
    for fname in os.listdir(dir_B):
        if fname.endswith('.jpg'):
            path = os.path.join(dir_B, fname)
            img_list_B.append(path)
    
    sorted(img_list_A)
    sorted(img_list_B)

    return img_list_A, img_list_B

class ImageTransform():
    """
    画像の前処理クラス．
    
    Attributes
    ----------
    mean : float
        画素値の平均
    std : float
        画素値の標準偏差
    """

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize((128, 128), Image.BICUBIC),  # リサイズ
            transforms.ToTensor(),                         # テンソルに変換
            transforms.Normalize(mean, std)                # 正規化
        ])
    
    def __call__(self, img):
        return self.data_transform(img)

class UnpairedDataset(data.Dataset):
    """
    画像のDatasetクラス．PyTorchのDatasetクラスを継承．

    Attributes
    ----------
    img_list_A : list
        集合Aの各画像へのパス
    img_list_B : list
        集合Bの各画像へのパス
    transform : torchvision.transforms
        画像の前処理クラス
    """

    def __init__(self, img_list_A, img_list_B, transform):
        self.img_list_A = img_list_A
        self.img_list_B = img_list_B
        self.size_A = len(self.img_list_A)  # 集合Aの画像の数
        self.size_B = len(self.img_list_B)  # 集合Bの画像の数
        self.transform = transform

    def __len__(self):
        """少ない方の集合の画像の枚数を返す"""
        return min(self.size_A, self.size_B)

    def __getitem__(self, index):
        index_A = index % self.size_A
        path_A = self.img_list_A[index_A]

        # 集合Bの画像はランダムに選択
        index_B = random.randint(0, self.size_B - 1)
        path_B = self.img_list_B[index_B]

        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        
        # 変換
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B, 'path_A': path_A, 'path_B': path_B}

def imshow(img):
    """
    torch.Tensorを画像として表示する関数．

    Parameters
    ----------
    img : torch.Tensor[batch, channels, height, width]
        画像
    """
    img = img.squeeze()          # batchの次元を潰す
    img_np = img.numpy()         # NumPy配列に変換
    img_np = 0.5 * (img_np + 1)  # 正規化を元に戻す，[-1, 1] -> [0, 1]
    plt.imshow(np.transpose(img_np, (1, 2, 0)))  # [c, h, w] -> [h, w, c]
