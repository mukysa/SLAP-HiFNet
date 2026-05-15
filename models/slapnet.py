# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from models.swinT import SWIN_VARIANTS  # Swin backbone variants used in this project / 本项目使用的 Swin 主干变体
from models.head import SegHead

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


# ========== Utility modules / 基础功能模块 ========== #
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
        V = self.conv_value(y).view(B, C, -1)             # Value features from y / 来自 y 的 value 特征，[B,C,HW]
        Q = self.conv_query(x).view(B, C, -1)             # Query features from x / 来自 x 的 query 特征
        K = self.conv_key(y).view(B, C, -1)               # Key features from y / 来自 y 的 key 特征

        # Mean centering and L2 normalization stabilize the attention scores.
        # 去均值与 L2 标准化用于稳定注意力相似度，避免幅度过大。
        Q = Q - Q.mean(dim=2, keepdim=True);  Q = F.normalize(Q, dim=1, eps=1e-6)
        K = K - K.mean(dim=2, keepdim=True);  K = F.normalize(K, dim=1, eps=1e-6)

        # Spatial affinity with temperature scaling.
        # 带温度系数的空间相似度矩阵。
        sim = torch.bmm(Q.transpose(1, 2), K) / max(self.tau, 1e-3)  # [B,HW,HW]
        A = self.softmax(sim)

        out = torch.bmm(A, V.transpose(1, 2)).transpose(1, 2)        # Attention-propagated features / 注意力传播后的特征，[B,C,HW]
        out = out.view(B, C, H, W)
        return out


class LFM(nn.Module):
    def __init__(self, channels_trans, channels_cnn, channels_fuse, tau=0.7, bottleneck=True):
        super(LFM, self).__init__()
        self.tau = tau
        mid = channels_fuse // 2 if bottleneck else channels_fuse

        # Channel projection and optional bottleneck reduction.
        # 通道投影，并通过可选瓶颈层降低注意力分支的计算量。
        self.conv_trans = nn.Conv2d(channels_trans, channels_fuse, 1, bias=False)
        self.conv_cnn   = nn.Conv2d(channels_cnn,  channels_fuse, 1, bias=False)
        self.red_t = nn.Conv2d(channels_fuse, mid, 1, bias=False)
        self.red_c = nn.Conv2d(channels_fuse, mid, 1, bias=False)

        # Bidirectional CAFM attention branches.
        # 双向 CAFM 注意力分支。
        self._CAFM1 = _CAFM(channels=mid, tau=self.tau)
        self._CAFM2 = _CAFM(channels=mid, tau=self.tau)

        # Project the concatenated attention features back to the fusion dimension.
        # 将拼接后的注意力特征投影回融合通道数。
        self.attn_proj = nn.Conv2d(2*mid, channels_fuse, 1, bias=False)

        # Concatenation-based baseline fusion branch.
        # 基于拼接的基础融合分支。
        self.base_cat = ConvBNReLU(2*channels_fuse, channels_fuse, k=1, s=1, p=0)

        # Learnable attention weight initialized from zero.
        # 可学习注意力权重，从 0 初始化，使注意力分支逐步参与融合。
        self.lambda_attn = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        # Align spatial size before fusion.
        # 融合前对齐空间尺寸。
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], **up_kwargs)

        # Project both modalities to the same fusion dimension.
        # 将两种模态投影到相同的融合通道数。
        xt = self.conv_trans(x); yt = self.conv_cnn(y)
        xt_r = self.red_t(xt);   yt_r = self.red_c(yt)

        # Normalization and temperature scaling are implemented inside _CAFM.
        # 标准化与温度缩放在 _CAFM 内部完成。

        # Bidirectional attention-enhanced feature interaction.
        # 双向注意力增强的特征交互。
        l2g = xt_r + self._CAFM1(xt_r, yt_r).contiguous()
        g2l = yt_r + self._CAFM2(yt_r, xt_r).contiguous()
        attn = self.attn_proj(torch.cat([l2g, g2l], dim=1))

        # Baseline concatenation branch.
        # 基础拼接融合分支。
        base = self.base_cat(torch.cat([xt, yt], dim=1))

        # Mixture of baseline fusion and attention-enhanced fusion.
        # 将基础融合结果与注意力增强结果进行可学习混合。
        lam = torch.sigmoid(self.lambda_attn)     # Learnable scalar in (0,1) / 映射到 (0,1) 的可学习系数
        out = base + lam * attn
        return out


class GFM(nn.Module):
    def __init__(self, channels_trans, channels_cnn, channels_fuse, residual=True):
        super(GFM, self).__init__()
        self.residual = residual
        self.conv_trans = nn.Conv2d(channels_trans, channels_fuse, kernel_size=1, bias=False)
        self.conv_cnn  = nn.Conv2d(channels_cnn,  channels_fuse, kernel_size=1, bias=False)

        # Lightweight gate: DW 3x3 + PW 1x1. Zero bias makes the initial gate close to 0.5.
        # 轻量门控：深度卷积 3×3 + 点卷积 1×1。偏置置零使初始门控值接近 0.5。
        self.gate_dw = nn.Conv2d(2*channels_fuse, 2*channels_fuse, 3, 1, 1,
                                 groups=2*channels_fuse, bias=False)
        self.gate_pw = nn.Conv2d(2*channels_fuse, channels_fuse, 1, 1, 0, bias=True)
        nn.init.zeros_(self.gate_pw.bias)  # Initialize gate around 0.5 / 将门控初始化在约 0.5 附近

        if residual:
            self.res_conv = nn.Conv2d(channels_trans + channels_cnn, channels_fuse, 1, 1, 0, bias=False)

    def forward(self, x, y):
        # Align spatial size before fusion.
        # 融合前对齐空间尺寸。
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], **up_kwargs)

        x = self.conv_trans(x); y = self.conv_cnn(y)
        xy = torch.cat([x, y], dim=1)
        g  = torch.sigmoid(self.gate_pw(self.gate_dw(xy)))   # Spatial-channel gate / 空间-通道门控，[B,C,H,W]
        out = g * x + (1.0 - g) * y

        if self.residual:
            out = out + self.res_conv(torch.cat([x, y], dim=1))  # Residual fusion on projected features / 对投影后的特征进行残差融合
        return out


# ========== Main network / 主网络 ========== #
class SLAPNet(nn.Module):
    """
    Input: x (B, 9, H, W), where the first 3 channels are SAR and the remaining 6 channels are AMSR2.
    输入：x (B, 9, H, W)，其中前 3 个通道为 SAR，后 6 个通道为 AMSR2。

    Backbone: two Swin Transformer branches are used separately for SAR and AMSR2 feature extraction.
    主干：使用两个 Swin Transformer 分支分别提取 SAR 与 AMSR2 多尺度特征。

    Fusion: the shallow two scales use GFM, and the deeper two scales use LFM with bidirectional CAFM attention.
    融合：浅层两个尺度采用 GFM，深层两个尺度采用包含双向 CAFM 注意力的 LFM。

    Decoding: the four fused scales are decoded by task-specific SegHead modules.
    解码：四个融合尺度分别送入任务专用的 SegHead 解码器。

    Output: {'label_80_sic':[B,nclass,H,W], 'label_manual':[B,2,H,W]} when both heads are enabled.
    输出：当两个任务头均启用时，输出 {'label_80_sic':[B,nclass,H,W], 'label_manual':[B,2,H,W]}。
    """
    def __init__(self, options):
        super().__init__()
        self.opt = options
        self.nclass = options.get('nclass', 11)
        self.use_iw_head = options.get('use_iw_head', True)

        # Input channel settings and modality slicing.
        # 输入通道设置与模态切片配置。
        self.total_in = options.get('in_channels', 9)
        self.sar_in   = options.get('sar_in_channels', 3)     # SAR channels placed first / 前置的 SAR 通道数
        self.amsr_in  = options.get('amsr_in_channels', 6)    # AMSR2 channels placed after SAR / 位于 SAR 之后的 AMSR2 通道数
        assert self.total_in == self.sar_in + self.amsr_in, "in_channels ≠ sar+amsr"

        # Swin backbones for the two modality branches.
        # 两个模态分支使用的 Swin 主干。
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
            aux=False, head='none', edge_aux=False, in_chans=self.sar_in  # SAR input channels / SAR 输入通道数
        )
        amsr_model = amsr_builder(
            nclass=self.nclass, img_size=img_size, pretrained=pretrained,
            aux=False, head='none', edge_aux=False, in_chans=self.amsr_in  # AMSR2 input channels / AMSR2 输入通道数
        )

        self.sar_swin  = sar_model.backbone
        self.amsr_swin = amsr_model.backbone

        sar_dims = getattr(sar_model, 'head_dim', None)
        amsr_dims = getattr(amsr_model, 'head_dim', None)
        if sar_dims is None or amsr_dims is None:
            raise RuntimeError("请确保 SWIN_VARIANTS 返回的模型含 head_dim（多尺度通道列表）。")

        # Project multi-scale features to a shared fusion dimension.
        # 将多尺度特征投影到统一的融合通道数。
        self.fuse_ch = options.get('fusion_channels', 128)    # Fusion channel width / 融合通道数
        # SAR feature alignment.
        # SAR 特征通道对齐。
        self.align_u1 = ConvBNReLU(sar_dims[-4], self.fuse_ch, 1, 1, 0)
        self.align_u2 = ConvBNReLU(sar_dims[-3], self.fuse_ch, 1, 1, 0)
        self.align_u3 = ConvBNReLU(sar_dims[-2], self.fuse_ch, 1, 1, 0)
        self.align_u4 = ConvBNReLU(sar_dims[-1], self.fuse_ch, 1, 1, 0)
        # AMSR2 feature alignment.
        # AMSR2 特征通道对齐。
        self.align_s1 = ConvBNReLU(amsr_dims[-4], self.fuse_ch, 1, 1, 0)
        self.align_s2 = ConvBNReLU(amsr_dims[-3], self.fuse_ch, 1, 1, 0)
        self.align_s3 = ConvBNReLU(amsr_dims[-2], self.fuse_ch, 1, 1, 0)
        self.align_s4 = ConvBNReLU(amsr_dims[-1], self.fuse_ch, 1, 1, 0)

        # Hierarchical fusion modules: GFM for shallow scales and LFM for deeper scales.
        # 分层融合模块：浅层尺度使用 GFM，深层尺度使用 LFM。
        self.f1 = GFM(channels_trans=self.fuse_ch, channels_cnn=self.fuse_ch, channels_fuse=self.fuse_ch, residual=True)
        self.f2 = GFM(channels_trans=self.fuse_ch, channels_cnn=self.fuse_ch, channels_fuse=self.fuse_ch, residual=True)
        self.f3 = LFM(channels_trans=self.fuse_ch, channels_cnn=self.fuse_ch, channels_fuse=self.fuse_ch)
        self.f4 = LFM(channels_trans=self.fuse_ch, channels_cnn=self.fuse_ch, channels_fuse=self.fuse_ch)

        # Task-specific decoders.
        # 任务专用解码器。
        # The four fused scales share the same channel width, so the SegHead input list is [fuse_ch]*4.
        # 四个融合尺度具有相同通道数，因此 SegHead 的输入通道列表为 [fuse_ch]*4。
        self.seghead_sic = SegHead(in_channels=[self.fuse_ch]*4, num_classes=self.nclass, in_index=[0,1,2,3])
        if self.use_iw_head:
            self.seghead_iw  = SegHead(in_channels=[self.fuse_ch]*4, num_classes=2, in_index=[0,1,2,3])
            
        # AMSR2-driven residual anchors for the SIC branch.
        # 面向 SIC 分支的 AMSR2 驱动残差锚点。
        self.sic_anchor1 = ConvBNReLU(self.fuse_ch, self.fuse_ch, k=1)
        self.sic_anchor2 = ConvBNReLU(self.fuse_ch, self.fuse_ch, k=1)
        self.sic_anchor3 = ConvBNReLU(self.fuse_ch, self.fuse_ch, k=1)
        self.sic_anchor4 = ConvBNReLU(self.fuse_ch, self.fuse_ch, k=1)
        # Learnable residual gates initialized near zero through sigmoid(-6)≈0.002.
        # 可学习残差门控，通过 sigmoid(-6)≈0.002 初始化为接近 0。
        self.sic_eps1 = nn.Parameter(torch.tensor(-6.0))
        self.sic_eps2 = nn.Parameter(torch.tensor(-6.0))
        self.sic_eps3 = nn.Parameter(torch.tensor(-6.0))
        self.sic_eps4 = nn.Parameter(torch.tensor(-6.0))

    def forward(self, x, mode='all', **kwargs):
        """
        x: (B, 9, H, W), with SAR in [0:3) and AMSR2 in [3:9) by default.
        x：(B, 9, H, W)，默认 [0:3) 为 SAR，[3:9) 为 AMSR2。
        """
        B, C, H, W = x.shape
        assert C == self.total_in, f"预期 {self.total_in} 通道，实际 {C}"

        # Split the input tensor into modality-specific inputs.
        # 将输入张量切分为不同模态的输入。
        x_sar  = x[:, :self.sar_in, :, :]   # SAR input / SAR 输入
        x_amsr = x[:, self.sar_in:, :, :]   # AMSR2 input / AMSR2 输入

        # Extract multi-scale features with the two Swin branches.
        # 使用两个 Swin 分支提取多尺度特征。
        u1, u2, u3, u4 = self.sar_swin(x_sar)     # SAR multi-scale features / SAR 多尺度特征
        s1, s2, s3, s4 = self.amsr_swin(x_amsr)   # AMSR2 multi-scale features / AMSR2 多尺度特征

        # Align channel dimensions before fusion.
        # 融合前对齐通道维度。
        u1 = self.align_u1(u1); u2 = self.align_u2(u2); u3 = self.align_u3(u3); u4 = self.align_u4(u4)
        s1 = self.align_s1(s1); s2 = self.align_s2(s2); s3 = self.align_s3(s3); s4 = self.align_s4(s4)

        # Multi-scale fusion: GFM for the first two scales and LFM for the last two scales.
        # 多尺度融合：前两个尺度使用 GFM，后两个尺度使用 LFM。
        p1 = self.f1(s1, u1)  # Fused scale 1 / 融合尺度 1
        p2 = self.f2(s2, u2)  # Fused scale 2 / 融合尺度 2
        p3 = self.f3(s3, u3)  # Fused scale 3 / 融合尺度 3
        p4 = self.f4(s4, u4)  # Fused scale 4 / 融合尺度 4
        
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
            y = F.interpolate(y, size=(H, W), **up_kwargs)  # Resize to the original input size / 上采样到原始输入尺寸
            output_dict['label_80_sic'] = [y]

        if self.use_iw_head and mode in ['iw_only', 'all']:
            z = self.seghead_iw(feat_list_iw)
            z = F.interpolate(z, size=(H, W), **up_kwargs)  # Resize to the original input size / 上采样到原始输入尺寸
            output_dict['label_manual'] = [z]

        return output_dict
