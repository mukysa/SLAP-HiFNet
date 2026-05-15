import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class PSPNet_DualHead(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.nclass = options['nclass']
        self.in_channels = options.get('in_channels', 3)
        self.encoder_name = options.get('encoder_name', 'resnet50')
        self.encoder_weights = options.get('encoder_weights', 'imagenet')
        self.use_iw_head = options.get('use_iw_head', True)

        # 构建 PSPNet 并移除原始 head
        self.encoder_decoder = smp.PSPNet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=1  # 暂用
        )
        self.encoder_decoder.segmentation_head = nn.Identity()

        feat_channels = self.encoder_decoder.encoder.out_channels[-1]

        # SIC Head
        self.head_sic = nn.Sequential(
            nn.Conv2d(feat_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, self.nclass, kernel_size=1)
        )

        # IW Head
        if self.use_iw_head:
            self.head_iw = nn.Sequential(
                nn.Conv2d(feat_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(64, 2, kernel_size=1)
            )


    def forward(self, x, mode='all', **kwargs):
        size = x.shape[2:]
        feat = self.encoder_decoder(x)  # 输出 shape: [B, 64, H, W]

        output_dict = {}

        if mode in ['sic_only', 'all']:
            out_sic = self.head_sic(feat)
            out_sic = nn.functional.interpolate(out_sic, size, mode='bilinear', align_corners=False)
            output_dict['label_80_sic'] = [out_sic]

        if self.use_iw_head and mode in ['iw_only', 'all']:
            out_iw = self.head_iw(feat)
            out_iw = nn.functional.interpolate(out_iw, size, mode='bilinear', align_corners=False)
            output_dict['label_manual'] = [out_iw]

        return output_dict
