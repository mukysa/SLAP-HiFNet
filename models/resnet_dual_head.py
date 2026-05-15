# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


class ResNet_DualHead(nn.Module):
    def __init__(self, options):
        super().__init__()
        encoder_name     = options.get('backbone', 'resnet50')      # resnet18/34/50 均可
        in_channels      = options.get('in_channels', 9)            # 3(SAR)+6(AMSR2)
        encoder_weights  = options.get('encoder_weights', 'imagenet')
        self.nclass      = options.get('nclass', 11)
        self.use_iw_head = options.get('use_iw_head', True)

        # ========== 从UNet骨干获取ResNet ==========
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1  # dummy
        )
        self.unet.decoder = nn.Identity()          # [MOD] 只用 encoder
        self.unet.segmentation_head = nn.Identity()

        # encoder 多尺度通道；最后一层通道作为双头输入
        enc_channels = getattr(self.unet.encoder, 'out_channels', [64,64,128,256,512])
        last_ch = enc_channels[-1]

        # ========== 双头 ==========
        self.head_sic = nn.Sequential(
            nn.Conv2d(last_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, self.nclass, kernel_size=1)
        )
        if self.use_iw_head:
            self.head_iw = nn.Sequential(
                nn.Conv2d(last_ch, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(64, 2, kernel_size=1)
            )

    def forward(self, x, mode='all', **kwargs):
        H, W = x.shape[-2], x.shape[-1]

        # [MOD] 直接取 encoder 多尺度特征
        feats = self.unet.encoder(x)        # list: [..., C3, C4, C5]
        f = feats[-1]                       # 最深层 (B, C5, H/32, W/32)

        out = {}
        if mode in ['sic_only', 'all']:
            y = self.head_sic(f)
            y = F.interpolate(y, size=(H, W), **up_kwargs)
            out['label_80_sic'] = [y]
        if self.use_iw_head and mode in ['iw_only', 'all']:
            z = self.head_iw(f)
            z = F.interpolate(z, size=(H, W), **up_kwargs)
            out['label_manual'] = [z]

        return out
