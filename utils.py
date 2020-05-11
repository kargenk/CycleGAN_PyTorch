#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

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
