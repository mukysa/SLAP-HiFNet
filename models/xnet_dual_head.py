import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder

class XNet_DualHead(nn.Module):
    def __init__(self, options):
        super(XNet_DualHead, self).__init__()
        self.in_channels = options.get('in_channels', 3)
        self.nclass = options.get('nclass', 11)
        self.use_iw_head = options.get('use_iw_head', True)
        self.depth = options.get('depth', 5)
        self.shrink_ratio = options.get('shrink_ratio', 2)

        # 初始化编码器
        encoder_name = options.get('backbone', 'resnet50')
        encoder_weights = options.get('encoder_weights', 'imagenet')  # or None
        self.encoder = get_encoder(
            name=encoder_name,
            in_channels=self.in_channels,
            depth=self.depth,
            weights=encoder_weights
        )

        encoder_channels = self.encoder.out_channels  # e.g. [64, 256, 512, 1024, 2048]
        decoder_out_channels = [encoder_channels[-1]]  # 初始化 decoder 输出通道列表

        # 构建 decoder（倒序）
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(encoder_channels) - 1)):
            in_ch = decoder_out_channels[-1] + encoder_channels[i]
            out_ch = encoder_channels[i]  // self.shrink_ratio  # shrink_ratio = 2
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
            self.decoder.append(block)
            decoder_out_channels.append(out_ch)
        decoder_out_channels = decoder_out_channels[::-1]

        # 最后的融合输出
        final_ch = decoder_out_channels[0] + encoder_channels[0]
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_ch, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # SIC Head
        self.head_sic = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, self.nclass, kernel_size=1)
        )

        # IW Head
        if self.use_iw_head:
            self.head_iw = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(64, 2, kernel_size=1)
            )

    def forward(self, x, mode='all', **kwargs):
        size = x.shape[2:]
        features = self.encoder(x)  # 5级特征

        out = features[-1]
        for i, decoder_block in enumerate(self.decoder):
            skip = features[-(i + 2)]
            if skip.shape[2:] != out.shape[2:]:
                skip = F.interpolate(skip, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = decoder_block(torch.cat([out, skip], dim=1))

        fused = torch.cat([
            out, 
            F.interpolate(features[0], size=out.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)
        fused = self.final_conv(fused)

        output_dict = {}
        if mode in ['sic_only', 'all']:
            out_sic = self.head_sic(fused)
            output_dict['label_80_sic'] = [F.interpolate(out_sic, size, mode='bilinear', align_corners=False)]
        if self.use_iw_head and mode in ['iw_only', 'all']:
            out_iw = self.head_iw(fused)
            output_dict['label_manual'] = [F.interpolate(out_iw, size, mode='bilinear', align_corners=False)]

        return output_dict
