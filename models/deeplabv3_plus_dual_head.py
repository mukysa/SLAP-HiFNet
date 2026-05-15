import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Plus_DualHead(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.in_channels = options.get('in_channels', 3)
        self.nclass = options['nclass']
        self.use_iw_head = options.get('use_iw_head', True)

        # 共享主干 + ASPP（解码器）
        self.backbone_name = options.get('backbone', 'resnet50')
        self.encoder_weights = options.get('encoder_weights', 'imagenet')  # 或 None

        # 构建 DeepLabV3+ 模型（仅使用 shared encoder + decoder）
        self.base_model = smp.DeepLabV3Plus(
            encoder_name=self.backbone_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=0  # 实际后续不使用此 head
        )
        
        # 移除 segmentation head
        self.base_model.segmentation_head = nn.Identity()

        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder

        # decoder 输出通道数（官方为256）
        decoder_out_channels = self.base_model.decoder.out_channels

        # SIC Head
        self.head_sic = nn.Sequential(
            nn.Conv2d(decoder_out_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, self.nclass, 1)
        )

        # IW Head
        if self.use_iw_head:
            self.head_iw = nn.Sequential(
                nn.Conv2d(decoder_out_channels, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(64, 2, 1)
            )


    def forward(self, x, mode='all', **kwargs):
        size = x.shape[2:]

        features = self.encoder(x)
        decoder_out = self.decoder(*features)

        output_dict = {}
        if mode in ['sic_only', 'all']:
            out_sic = self.head_sic(decoder_out)
            out_sic = nn.functional.interpolate(out_sic, size=size, mode='bilinear', align_corners=False)
            output_dict['label_80_sic'] = [out_sic]

        if self.use_iw_head and mode in ['iw_only', 'all']:
            out_iw = self.head_iw(decoder_out)
            out_iw = nn.functional.interpolate(out_iw, size=size, mode='bilinear', align_corners=False)
            output_dict['label_manual'] = [out_iw]

        return output_dict

