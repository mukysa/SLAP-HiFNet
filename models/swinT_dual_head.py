import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swinT import SWIN_VARIANTS  # 包含 'swin_tiny': swin_tiny_nband 等

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

def make_conv_head(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(64, out_ch, 1)
    )

class SwinT_DualHead(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.opt = options
        self.nclass = options['nclass']
        self.use_iw_head = options.get('use_iw_head', False)
        in_ch = options.get('in_channels', 9)
        img_size = options.get('sar_patch_size', 1024)

        # 选择变体
        name = options.get('backbone', 'swin_tiny')
        assert name in SWIN_VARIANTS, f'Unknown Swin variant: {name}'
        builder = SWIN_VARIANTS[name]

        # 构建 SwinT 模型（我们只取其中的 .backbone 与 .head_dim）
        swin_model = builder(
            nclass=self.nclass,
            img_size=img_size,
            pretrained=options.get('pretrained', True),
            aux=False,
            head='none',
            edge_aux=False
        )

        # 主干和最后输出通道
        self.backbone = swin_model.backbone
        decoder_dim = swin_model.head_dim[-1]

        # 任务分支
        self.head_sic = make_conv_head(decoder_dim, self.nclass)
        if self.use_iw_head:
            self.head_iw = make_conv_head(decoder_dim, 2)

    def forward(self, x, mode='all', **kwargs):
        size = x.shape[2:]
        feats = self.backbone(x)       # (p1, p2, p3, p4)
        feat = feats[-1]               # 最后一层特征

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
