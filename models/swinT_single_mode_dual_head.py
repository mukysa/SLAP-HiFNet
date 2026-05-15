# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swinT import SWIN_VARIANTS  # 如上定义，支持 in_chans

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

def make_conv_head(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(64, out_ch, 1)
    )

class _SwinT_DualHead_Base(nn.Module):
    def __init__(self, options, enc_in_ch):
        super().__init__()
        self.opt = options
        self.nclass = options['nclass']
        self.use_iw_head = options.get('use_iw_head', False)
        img_size = options.get('sar_patch_size', 1024)
        name = options.get('backbone', 'swin_tiny')
        assert name in SWIN_VARIANTS, f'Unknown Swin variant: {name}'
        builder = SWIN_VARIANTS[name]

        # 关键：直接把 in_chans 设为 3 或 6
        swin_model = builder(
            nclass=self.nclass,
            img_size=img_size,
            pretrained=options.get('pretrained', True),
            aux=False, head='none', edge_aux=False,
            in_chans=enc_in_ch
        )

        self.backbone = swin_model.backbone
        decoder_dim = swin_model.head_dim[-1]
        self.head_sic = make_conv_head(decoder_dim, self.nclass)
        if self.use_iw_head:
            self.head_iw = make_conv_head(decoder_dim, 2)

    # 子类实现：输入切片
    def _slice_input(self, x):
        raise NotImplementedError

    def forward(self, x, mode='all', **kwargs):
        size = x.shape[2:]
        x_in = self._slice_input(x)
        feats = self.backbone(x_in)   # (p1, p2, p3, p4)
        feat = feats[-1]

        out = {}
        if mode in ['sic_only', 'all']:
            y = self.head_sic(feat)
            y = F.interpolate(y, size=size, **up_kwargs)
            out['label_80_sic'] = [y]

        if self.use_iw_head and mode in ['iw_only', 'all']:
            z = self.head_iw(feat)
            z = F.interpolate(z, size=size, **up_kwargs)
            out['label_manual'] = [z]

        return out


class SwinT_DualHead_SAR(_SwinT_DualHead_Base):
    """仅用前3通道（SAR）"""
    def __init__(self, options):
        super().__init__(options, enc_in_ch=3)

    def _slice_input(self, x):
        assert x.shape[1] >= 3, f"SAR-only 需要>=3通道，得到 {x.shape[1]}"
        return x[:, :3, ...]


class SwinT_DualHead_AMSR2(_SwinT_DualHead_Base):
    """仅用后6通道（AMSR2）"""
    def __init__(self, options):
        super().__init__(options, enc_in_ch=6)

    def _slice_input(self, x):
        if x.shape[1] >= 9:
            return x[:, -6:, ...]
        elif x.shape[1] == 6:
            return x
        else:
            raise AssertionError(f"AMSR2-only 需要输入为 6 或 9 通道，得到 {x.shape[1]}")
