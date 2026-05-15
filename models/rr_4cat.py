# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from models.swinT import SWIN_VARIANTS  # 你的现有实现
from models.head import SegHead

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


# ========== 小组件 ========== #
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


# ========== 小组件（新增一个简单的 Cat 融合块） ========== #
class CatFuse(nn.Module):
    """cat 后用 1x1 投影到 fuse_ch；稳定廉价"""
    def __init__(self, ch):
        super().__init__()
        self.proj = ConvBNReLU(2 * ch, ch, k=1, s=1, p=0)
    def forward(self, u, s):
        if u.shape[-2:] != s.shape[-2:]:
            s = F.interpolate(s, size=u.shape[-2:], **up_kwargs)
        return self.proj(torch.cat([u, s], dim=1))



# ========== 主网络 ========== #
class RR4cat(nn.Module):
    """
    输入：x (B, 9, H, W)，前3为SAR，后6为AMSR2
    分支：UNet(encoder) 处理 SAR；Swin(backbone) 处理 AMSR2
    融合：P1/P2/P3/P4（四层，每层对称 softmax 空间门控 + SE 通道重标定）
    解码：四层直接送入 SegHead 聚合到最高分辨率
    输出：{'label_80_sic':[B,nclass,H,W], 'label_manual':[B,2,H,W]}
    """
    def __init__(self, options):
        super().__init__()
        self.opt = options
        self.nclass = options.get('nclass', 11)
        self.use_iw_head = options.get('use_iw_head', True)

        # ---- 输入通道与切片 ---- #
        self.total_in = options.get('in_channels', 9)
        self.sar_in   = options.get('sar_in_channels', 3)     # 前3
        self.amsr_in  = options.get('amsr_in_channels', 6)    # 后6
        assert self.total_in == self.sar_in + self.amsr_in, "in_channels ≠ sar+amsr"

        # ---- UNet 编码器（SAR）---- #
        sar_encoder = options.get('sar_encoder', 'resnet34')
        unet_weights = options.get('sar_encoder_weights', 'imagenet')  # SAR 3通道可用 'imagenet'
        self.unet_sar = smp.Unet(
            encoder_name=sar_encoder,
            encoder_weights=unet_weights,
            in_channels=self.sar_in,
            classes=1  # dummy
        )
        # 只用 encoder，多尺度特征
        self.unet_sar.decoder = nn.Identity()
        self.unet_sar.segmentation_head = nn.Identity()
        # 估计 UNet 多尺度通道（取后三层作为 P2/P3/P4）
        u_ch = getattr(self.unet_sar.encoder, 'out_channels', [64,64,128,256,512])

        # ---- Swin 主干（AMSR2，6通道）---- #
        amsr_encoder = options.get('amsr_encoder', sar_encoder)                  # [MOD] 与 SAR 同 backbone 为默认
        amsr_weights = options.get('amsr_encoder_weights', None)                 # [MOD] 默认 None；避免 6ch + imagenet 冲突
        self.unet_amsr = smp.Unet(
            encoder_name=amsr_encoder,
            encoder_weights=amsr_weights,
            in_channels=self.amsr_in,  # 6
            classes=1
        )
        self.unet_amsr.decoder = nn.Identity()
        self.unet_amsr.segmentation_head = nn.Identity()
        v_ch = getattr(self.unet_amsr.encoder, 'out_channels', [64,64,128,256,512])


        # ---- 通道对齐到同一维度（融合通道）---- #
        self.fuse_ch = options.get('fusion_channels', 128)
        
        # UNet (SAR)
        self.align_u1 = ConvBNReLU(u_ch[-4], self.fuse_ch, 1, 1, 0)
        self.align_u2 = ConvBNReLU(u_ch[-3], self.fuse_ch, 1, 1, 0)
        self.align_u3 = ConvBNReLU(u_ch[-2], self.fuse_ch, 1, 1, 0)
        self.align_u4 = ConvBNReLU(u_ch[-1], self.fuse_ch, 1, 1, 0)
        # Swin (AMSR2)
        self.align_s1 = ConvBNReLU(v_ch[-4], self.fuse_ch, 1, 1, 0)
        self.align_s2 = ConvBNReLU(v_ch[-3], self.fuse_ch, 1, 1, 0)
        self.align_s3 = ConvBNReLU(v_ch[-2], self.fuse_ch, 1, 1, 0)
        self.align_s4 = ConvBNReLU(v_ch[-1], self.fuse_ch, 1, 1, 0)

        # ---- 4层统一：由“基础门控”改为“cat+直接融合” ---- #
        self.f1 = CatFuse(self.fuse_ch)
        self.f2 = CatFuse(self.fuse_ch)
        self.f3 = CatFuse(self.fuse_ch)
        self.f4 = CatFuse(self.fuse_ch)

        # ---- 解码器 ---- #
        # 四层融合后直接进SegHead，四层通道相同 → [fuse_ch]*4
        self.seghead_sic = SegHead(in_channels=[self.fuse_ch]*4, num_classes=self.nclass, in_index=[0,1,2,3])
        if self.use_iw_head:
            self.seghead_iw  = SegHead(in_channels=[self.fuse_ch]*4, num_classes=2, in_index=[0,1,2,3])

    def forward(self, x, mode='all', **kwargs):
        """
        x: (B, 9, H, W)  [0:3)=SAR, [3:9)=AMSR2
        """
        B, C, H, W = x.shape
        assert C == self.total_in, f"预期 {self.total_in} 通道，实际 {C}"

        # --- 切分模态 --- #
        x_sar  = x[:, :self.sar_in, :, :]   # (B, 3, H, W)
        x_amsr = x[:, self.sar_in:, :, :]   # (B, 6, H, W)

        # --- UNet encoder（多尺度） --- #
        u_feats = self.unet_sar.encoder(x_sar)
        u1, u2, u3, u4 = u_feats[-4], u_feats[-3], u_feats[-2], u_feats[-1]

        # --- Swin backbone（多尺度，直接 6 通道输入） --- #
        s_feats = self.unet_amsr.encoder(x_amsr)
        s1, s2, s3, s4 = s_feats[-4], s_feats[-3], s_feats[-2], s_feats[-1]

        # --- 通道对齐 --- #
        u1 = self.align_u1(u1); u2 = self.align_u2(u2); u3 = self.align_u3(u3); u4 = self.align_u4(u4)
        s1 = self.align_s1(s1); s2 = self.align_s2(s2); s3 = self.align_s3(s3); s4 = self.align_s4(s4)

        # --- 多尺度 cat 直接融合 --- #
        p1 = self.f1(u1, s1)          # (B, C, H/2,  W/2)
        p2 = self.f2(u2, s2)          # (B, C, H/4,  W/4)
        p3 = self.f3(u3, s3)          # (B, C, H/8,  W/8)
        p4 = self.f4(u4, s4)          # (B, C, H/16, W/16)
        
        feat_list = [p1, p2, p3, p4]
        
        output_dict = {}
        if mode in ['sic_only', 'all']:
            y = self.seghead_sic(feat_list)
            y = F.interpolate(y, size=(H, W), **up_kwargs)  # 统一到原图大小
            output_dict['label_80_sic'] = [y]

        if self.use_iw_head and mode in ['iw_only', 'all']:
            z = self.seghead_iw(feat_list)
            z = F.interpolate(z, size=(H, W), **up_kwargs)
            output_dict['label_manual'] = [z]

        return output_dict
