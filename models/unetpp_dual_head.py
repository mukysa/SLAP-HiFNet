import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetPP_DualHead(nn.Module):
    def __init__(self, options):
        super().__init__()
        encoder_name = options.get('backbone', 'resnet34')  # 默认 backbone
        in_channels = options.get('in_channels', 3)
        encoder_weights = options.get('encoder_weights', 'imagenet')  # 或 None
        decoder_channels = options.get('decoder_channels', [256, 128, 64, 64, 64])
        self.nclass = options.get('nclass', 11)
        self.use_iw_head = options.get('use_iw_head', True)

        # 主干为 Unet++
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            classes=1,  # dummy
            decoder_attention_type=None  # 可选: 'scse'
        )
        
        decoder_out_channels = decoder_channels[-1]  # 自动获取最后一层输出通道
        
        # 删除 smp 自带 head
        self.backbone.segmentation_head = nn.Identity()

        # SIC Head
        self.head_sic = nn.Sequential(
            nn.Conv2d(decoder_out_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.nclass, kernel_size=1)
        )

        # IW Head（2类）
        if self.use_iw_head:
            self.head_iw = nn.Sequential(
                nn.Conv2d(decoder_out_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 2, kernel_size=1)
            )

    def forward(self, x, mode='all', **kwargs):
        size = x.shape[2:]
        features = self.backbone.decoder(*self.backbone.encoder(x))  # (B, 64, H, W)
        output_dict = {}

        if mode in ['sic_only', 'all']:
            out_sic = self.head_sic(features)
            output_dict['label_80_sic'] = [nn.functional.interpolate(out_sic, size, mode='bilinear', align_corners=False)]

        if self.use_iw_head and mode in ['iw_only', 'all']:
            out_iw = self.head_iw(features)
            output_dict['label_manual'] = [nn.functional.interpolate(out_iw, size, mode='bilinear', align_corners=False)]

        return output_dict
