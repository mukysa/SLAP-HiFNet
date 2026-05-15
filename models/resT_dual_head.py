import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resT import REST_VARIANTS  # {'rest_tiny': ..., ...}

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


def make_conv_head(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(64, out_ch, 1)
    )


class ResT_DualHead(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.opt = options
        self.nclass = options['nclass']
        self.use_iw_head = options.get('use_iw_head', False)
        in_ch = options.get('in_channels', 9)
        img_size = options.get('sar_patch_size', 1024)

        # 选择变体
        name = options.get('backbone', 'rest_tiny')
        assert name in REST_VARIANTS, f'Unknown Rest variant: {name}'
        builder = REST_VARIANTS[name]

        # 构建模型（不使用主干的 decode_head）
        rest_model = builder(
            nclass=self.nclass, nband=in_ch,
            pretrained=options.get('pretrained', True),
            aux=False, head='none', edge_aux=False
        )

        self.backbone = rest_model.backbone
        decoder_dim = rest_model.head_dim[-1]  # e.g., 512 或 768

        # 两个任务头（SIC 和 IW）
        self.head_sic = make_conv_head(decoder_dim, self.nclass)
        if self.use_iw_head:
            self.head_iw = make_conv_head(decoder_dim, 2)

    def forward(self, x, mode='all', **kwargs):
        size = x.shape[2:]
        feats = self.backbone(x)
        feat = feats[-1]

        out = {}

        if mode in ['all', 'sic_only']:
            y = self.head_sic(feat)
            y = F.interpolate(y, size=size, **up_kwargs)
            out['label_80_sic'] = [y]

        if self.use_iw_head and mode in ['all', 'iw_only']:
            z = self.head_iw(feat)
            z = F.interpolate(z, size=size, **up_kwargs)
            out['label_manual'] = [z]

        return out
