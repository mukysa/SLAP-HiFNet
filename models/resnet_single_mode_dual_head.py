# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


class _ResNet_DualHead_Base(nn.Module):
    """共享的双头解码结构；子类只需实现 _slice_input(x) 和 _init_encoder()."""
    def __init__(self, options, enc_in_channels, backbone_default='resnet50'):
        super().__init__()
        encoder_name    = options.get('backbone', backbone_default)    # resnet18/34/50
        encoder_weights = options.get('encoder_weights', 'imagenet')
        self.nclass     = options.get('nclass', 11)
        self.use_iw_head= options.get('use_iw_head', True)

        # 仅构建 encoder（Unet 的 decoder/seg_head 置空）
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=enc_in_channels,   # 由子类决定：3 或 6
            classes=1                      # dummy
        )
        self.unet.decoder = nn.Identity()
        self.unet.segmentation_head = nn.Identity()

        enc_channels = getattr(self.unet.encoder, 'out_channels', [64,64,128,256,512])
        last_ch = enc_channels[-1]

        # 双头（SIC 11 类；IW 2 类）
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

    # --- 子类需要实现 ---
    def _slice_input(self, x):
        raise NotImplementedError

    def forward(self, x, mode='all', **kwargs):
        H, W = x.shape[-2], x.shape[-1]
        x_in = self._slice_input(x)                 # 模态切片

        feats = self.unet.encoder(x_in)             # [..., C3, C4, C5]
        f = feats[-1]                               # (B, C5, H/32, W/32)

        out = {}
        if mode in ['sic_only', 'all']:
            y = self.head_sic(f)
            y = F.interpolate(y, size=(H, W), **up_kwargs)
            out['label_80_sic'] = [y]
        if getattr(self, 'use_iw_head', True) and mode in ['iw_only', 'all']:
            z = self.head_iw(f)
            z = F.interpolate(z, size=(H, W), **up_kwargs)
            out['label_manual'] = [z]
        return out


class ResNet_DualHead_SAR(_ResNet_DualHead_Base):
    """
    仅使用前3通道（SAR）进行编码；forward 仍接收 9 通道输入并内部切片 x[:, :3, ...]
    """
    def __init__(self, options):
        super().__init__(options, enc_in_channels=3, backbone_default=options.get('backbone', 'resnet50'))

    def _slice_input(self, x):
        # 期望输入至少3通道
        assert x.shape[1] >= 3, f"SAR-only 期望>=3通道，得到 {x.shape[1]}"
        return x[:, :3, ...]


class ResNet_DualHead_AMSR2(_ResNet_DualHead_Base):
    """
    仅使用后6通道（AMSR2）进行编码；forward 仍接收 9 通道输入并内部切片 x[:, -6:, ...]
    """
    def __init__(self, options):
        super().__init__(options, enc_in_channels=6, backbone_default=options.get('backbone', 'resnet50'))

    def _slice_input(self, x):
        # 期望输入至少9通道（以便后6通道为 AMSR2）；若仅有6通道也可兼容
        if x.shape[1] >= 9:
            return x[:, -6:, ...]
        elif x.shape[1] == 6:
            return x
        else:
            raise AssertionError(f"AMSR2-only 期望输入为 6 或 9 通道，得到 {x.shape[1]}")
