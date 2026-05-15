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


class _CAFM(nn.Module):
    def __init__(self, channels, tau=0.7):
        super(_CAFM, self).__init__()
        self.conv_value = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv_query = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv_key   = nn.Conv2d(channels, channels, 1, bias=False)
        self.tau = tau
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, y):
        B, C, H, W = x.shape
        V = self.conv_value(y).view(B, C, -1)             # [B,C,HW]
        Q = self.conv_query(x).view(B, C, -1)
        K = self.conv_key(y).view(B, C, -1)

        # 去均值 + L2 标准化，避免幅度爆炸
        Q = Q - Q.mean(dim=2, keepdim=True);  Q = F.normalize(Q, dim=1, eps=1e-6)
        K = K - K.mean(dim=2, keepdim=True);  K = F.normalize(K, dim=1, eps=1e-6)

        # 相似度 & 温度
        sim = torch.bmm(Q.transpose(1, 2), K) / max(self.tau, 1e-3)  # [B,HW,HW]
        A = self.softmax(sim)

        out = torch.bmm(A, V.transpose(1, 2)).transpose(1, 2)        # [B,C,HW]
        out = out.view(B, C, H, W)
        return out


class LFM(nn.Module):
    def __init__(self, channels_trans, channels_cnn, channels_fuse, tau=0.7, bottleneck=True):
        super(LFM, self).__init__()
        self.tau = tau
        mid = channels_fuse // 2 if bottleneck else channels_fuse

        # 对齐 + 降维
        self.conv_trans = nn.Conv2d(channels_trans, channels_fuse, 1, bias=False)
        self.conv_cnn   = nn.Conv2d(channels_cnn,  channels_fuse, 1, bias=False)
        self.red_t = nn.Conv2d(channels_fuse, mid, 1, bias=False)
        self.red_c = nn.Conv2d(channels_fuse, mid, 1, bias=False)

        # 注意力支路（对称）
        self._CAFM1 = _CAFM(channels=mid, tau=self.tau)
        self._CAFM2 = _CAFM(channels=mid, tau=self.tau)

        # 注意力输出回到 fuse_ch
        self.attn_proj = nn.Conv2d(2*mid, channels_fuse, 1, bias=False)

        # Cat 主路（安全基线）
        self.base_cat = ConvBNReLU(2*channels_fuse, channels_fuse, k=1, s=1, p=0)

        # 可学习系数 λ（从 0 起步，退火生长）
        self.lambda_attn = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        # 尺寸对齐
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], **up_kwargs)

        # 对齐+降维
        xt = self.conv_trans(x); yt = self.conv_cnn(y)
        xt_r = self.red_t(xt);   yt_r = self.red_c(yt)

        # -------- 标准化 & 温度（放进 _CAFM）--------
        # 我们在 _CAFM 里做 normalize/temperature

        # 注意力支路
        l2g = xt_r + self._CAFM1(xt_r, yt_r).contiguous()
        g2l = yt_r + self._CAFM2(yt_r, xt_r).contiguous()
        attn = self.attn_proj(torch.cat([l2g, g2l], dim=1))

        # Cat 主路
        base = self.base_cat(torch.cat([xt, yt], dim=1))

        # MoE-λ 混合
        lam = torch.sigmoid(self.lambda_attn)     # 平滑且可学习到(0,1)
        out = base + lam * attn
        return out


class GFM(nn.Module):
    def __init__(self, channels_trans, channels_cnn, channels_fuse, residual=True):
        super(GFM, self).__init__()
        self.residual = residual
        self.conv_trans = nn.Conv2d(channels_trans, channels_fuse, kernel_size=1, bias=False)
        self.conv_cnn  = nn.Conv2d(channels_cnn,  channels_fuse, kernel_size=1, bias=False)

        # 轻量门控：DW 3x3 + 1x1，bias=0 使 g≈0.5，从 Cat 起步
        self.gate_dw = nn.Conv2d(2*channels_fuse, 2*channels_fuse, 3, 1, 1,
                                 groups=2*channels_fuse, bias=False)
        self.gate_pw = nn.Conv2d(2*channels_fuse, channels_fuse, 1, 1, 0, bias=True)
        nn.init.zeros_(self.gate_pw.bias)  # 关键：g≈0.5

        if residual:
            self.res_conv = nn.Conv2d(channels_trans + channels_cnn, channels_fuse, 1, 1, 0, bias=False)

    def forward(self, x, y):
        # 尺寸对齐
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], **up_kwargs)

        x = self.conv_trans(x); y = self.conv_cnn(y)
        xy = torch.cat([x, y], dim=1)
        g  = torch.sigmoid(self.gate_pw(self.gate_dw(xy)))   # [B,C,H,W]
        out = g * x + (1.0 - g) * y

        if self.residual:
            out = out + self.res_conv(torch.cat([x, y], dim=1))  # 需要可加成时再放开
        return out


# ========== 主网络 ========== #
class SLAPNet(nn.Module):
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

        # ---- Swin 主干（两支对称）---- #
        img_size   = options.get('sar_patch_size', 1024)
        pretrained = options.get('pretrained', True) 
        swin_sar_name = options.get('swin_sar_backbone', 'swin_tiny')
        swin_amsr_name = options.get('swin_amsr_backbone', 'swin_tiny')

        assert swin_sar_name in SWIN_VARIANTS, f'Unknown Swin: {swin_sar_name}'
        assert swin_amsr_name in SWIN_VARIANTS, f'Unknown Swin: {swin_amsr_name}'

        sar_builder = SWIN_VARIANTS[swin_sar_name]
        amsr_builder = SWIN_VARIANTS[swin_amsr_name]

        sar_model = sar_builder(
            nclass=self.nclass, img_size=img_size, pretrained=pretrained,
            aux=False, head='none', edge_aux=False, in_chans=self.sar_in  # 3ch
        )
        amsr_model = amsr_builder(
            nclass=self.nclass, img_size=img_size, pretrained=pretrained,
            aux=False, head='none', edge_aux=False, in_chans=self.amsr_in  # 6ch
        )

        self.sar_swin  = sar_model.backbone
        self.amsr_swin = amsr_model.backbone

        sar_dims = getattr(sar_model, 'head_dim', None)
        amsr_dims = getattr(amsr_model, 'head_dim', None)
        if sar_dims is None or amsr_dims is None:
            raise RuntimeError("请确保 SWIN_VARIANTS 返回的模型含 head_dim（多尺度通道列表）。")

        # ---- 通道对齐到同一维度（融合通道）---- #
        self.fuse_ch = options.get('fusion_channels', 128)    # [MOD]
        # SAR 对齐
        self.align_u1 = ConvBNReLU(sar_dims[-4], self.fuse_ch, 1, 1, 0)
        self.align_u2 = ConvBNReLU(sar_dims[-3], self.fuse_ch, 1, 1, 0)
        self.align_u3 = ConvBNReLU(sar_dims[-2], self.fuse_ch, 1, 1, 0)
        self.align_u4 = ConvBNReLU(sar_dims[-1], self.fuse_ch, 1, 1, 0)
        # AMSR2 对齐
        self.align_s1 = ConvBNReLU(amsr_dims[-4], self.fuse_ch, 1, 1, 0)
        self.align_s2 = ConvBNReLU(amsr_dims[-3], self.fuse_ch, 1, 1, 0)
        self.align_s3 = ConvBNReLU(amsr_dims[-2], self.fuse_ch, 1, 1, 0)
        self.align_s4 = ConvBNReLU(amsr_dims[-1], self.fuse_ch, 1, 1, 0)

        # ---- 替换前两层为 LAFM，后两层为 CAFM ---- #
        self.f1 = GFM(channels_trans=self.fuse_ch, channels_cnn=self.fuse_ch, channels_fuse=self.fuse_ch, residual=True)
        self.f2 = GFM(channels_trans=self.fuse_ch, channels_cnn=self.fuse_ch, channels_fuse=self.fuse_ch, residual=True)
        self.f3 = LFM(channels_trans=self.fuse_ch, channels_cnn=self.fuse_ch, channels_fuse=self.fuse_ch)
        self.f4 = LFM(channels_trans=self.fuse_ch, channels_cnn=self.fuse_ch, channels_fuse=self.fuse_ch)

        # ---- 解码器 ---- #
        # 四层融合后直接进SegHead，四层通道相同 → [fuse_ch]*4
        self.seghead_sic = SegHead(in_channels=[self.fuse_ch]*4, num_classes=self.nclass, in_index=[0,1,2,3])
        if self.use_iw_head:
            self.seghead_iw  = SegHead(in_channels=[self.fuse_ch]*4, num_classes=2, in_index=[0,1,2,3])
            
        # after align_s*, 在 DriFTNet.__init__ 末尾加
        self.sic_anchor1 = ConvBNReLU(self.fuse_ch, self.fuse_ch, k=1)
        self.sic_anchor2 = ConvBNReLU(self.fuse_ch, self.fuse_ch, k=1)
        self.sic_anchor3 = ConvBNReLU(self.fuse_ch, self.fuse_ch, k=1)
        self.sic_anchor4 = ConvBNReLU(self.fuse_ch, self.fuse_ch, k=1)
        # 可学习门，先几乎为 0（sigmoid(-6)≈0.002）
        self.sic_eps1 = nn.Parameter(torch.tensor(-6.0))
        self.sic_eps2 = nn.Parameter(torch.tensor(-6.0))
        self.sic_eps3 = nn.Parameter(torch.tensor(-6.0))
        self.sic_eps4 = nn.Parameter(torch.tensor(-6.0))

    def forward(self, x, mode='all', **kwargs):
        """
        x: (B, 9, H, W)  [0:3)=SAR, [3:9)=AMSR2
        """
        B, C, H, W = x.shape
        assert C == self.total_in, f"预期 {self.total_in} 通道，实际 {C}"

        # --- 切分模态 --- #
        x_sar  = x[:, :self.sar_in, :, :]   # (B, 3, H, W)
        x_amsr = x[:, self.sar_in:, :, :]   # (B, 6, H, W)

        # --- Swin backbone（多尺度） --- #
        u1, u2, u3, u4 = self.sar_swin(x_sar)     # SAR 3ch
        s1, s2, s3, s4 = self.amsr_swin(x_amsr)   # AMSR2 6ch

        # --- 通道对齐 --- #
        u1 = self.align_u1(u1); u2 = self.align_u2(u2); u3 = self.align_u3(u3); u4 = self.align_u4(u4)
        s1 = self.align_s1(s1); s2 = self.align_s2(s2); s3 = self.align_s3(s3); s4 = self.align_s4(s4)

        # --- 多尺度融合：前两层用 LAFM，后两层用 CAFM --- #
        p1 = self.f1(s1, u1)  # (B, C, H/2,  W/2)
        p2 = self.f2(s2, u2)  # (B, C, H/4,  W/4)
        p3 = self.f3(s3, u3)  # (B, C, H/8,  W/8)
        p4 = self.f4(s4, u4)  # (B, C, H/16, W/16)    
        
        e1 = torch.sigmoid(self.sic_eps1); e2 = torch.sigmoid(self.sic_eps2)
        e3 = torch.sigmoid(self.sic_eps3); e4 = torch.sigmoid(self.sic_eps4)
        p1_sic = p1 + e1 * self.sic_anchor1(s1)
        p2_sic = p2 + e2 * self.sic_anchor2(s2)
        p3_sic = p3 + e3 * self.sic_anchor3(s3)
        p4_sic = p4 + e4 * self.sic_anchor4(s4)

        feat_list_sic = [p1_sic, p2_sic, p3_sic, p4_sic]
        feat_list_iw  = [p1,     p2,     p3,     p4    ]
        
        output_dict = {}
        if mode in ['sic_only', 'all']:
            y = self.seghead_sic(feat_list_sic)
            y = F.interpolate(y, size=(H, W), **up_kwargs)  # 统一到原图大小
            output_dict['label_80_sic'] = [y]

        if self.use_iw_head and mode in ['iw_only', 'all']:
            z = self.seghead_iw(feat_list_iw)
            z = F.interpolate(z, size=(H, W), **up_kwargs)
            output_dict['label_manual'] = [z]

        return output_dict
