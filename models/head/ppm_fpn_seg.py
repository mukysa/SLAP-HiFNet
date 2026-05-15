# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class PPM(nn.Module):
    """ Pyramid Pooling on top feature (like PSPNet). """
    def __init__(self, in_ch, out_ch, bins=(1, 2, 3, 6)):
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for b in bins
        ])
        self.project = ConvBNReLU(in_ch + len(bins)*out_ch, out_ch, k=1, s=1, p=0)

    def forward(self, x):
        H, W = x.shape[-2:]
        pyr = [x]
        for path in self.paths:
            y = path(x)
            y = F.interpolate(y, size=(H, W), **up_kwargs)
            pyr.append(y)
        return self.project(torch.cat(pyr, dim=1))

class ReFuseBlock(nn.Module):
    """
    解码端再融合：把当前解码特征 f 与同层的 (u,s) 再做一次门控混合。
    产生每像素 3 路权重 α,β,γ = softmax，输出 α*f + β*u + γ*s。
    """
    def __init__(self, ch):
        super().__init__()
        self.proj_f = ConvBNReLU(ch, ch, k=1, s=1, p=0)
        self.proj_u = ConvBNReLU(ch, ch, k=1, s=1, p=0)
        self.proj_s = ConvBNReLU(ch, ch, k=1, s=1, p=0)
        self.gate   = nn.Conv2d(3*ch, 3, 1, bias=True)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, f, u, s):
        fu = self.proj_u(u); fs = self.proj_s(s); ff = self.proj_f(f)
        logits = self.gate(torch.cat([ff, fu, fs], dim=1))           # (B,3,H,W)
        w = torch.softmax(logits, dim=1)                              # (B,3,H,W)
        return w[:,0:1]*ff + w[:,1:2]*fu + w[:,2:3]*fs

class SegHeadPlus(nn.Module):
    """
    Drop-in 解码头：输入四层同通道特征 [p1,p2,p3,p4]（p1最高分辨率），
    可选在每层引入同尺度的 (u_i, s_i) 做 Re-Fuse；输出一个 logits。
    约定：in_channels = [C,C,C,C]
    """
    def __init__(self,
                 in_channels,       # e.g. [128,128,128,128]
                 num_classes,
                 ppm_bins=(1,2,3,6),
                 use_refuse=True,
                 lateral_ch=None):
        super().__init__()
        assert len(in_channels) == 4
        c1,c2,c3,c4 = in_channels
        C = c1
        lat_ch = lateral_ch or C

        # top: PPM on p4
        self.ppm = PPM(C, C, bins=ppm_bins)

        # lateral projections (P4..P1)
        self.l4 = ConvBNReLU(c4, lat_ch, k=1, s=1, p=0)
        self.l3 = ConvBNReLU(c3, lat_ch, k=1, s=1, p=0)
        self.l2 = ConvBNReLU(c2, lat_ch, k=1, s=1, p=0)
        self.l1 = ConvBNReLU(c1, lat_ch, k=1, s=1, p=0)

        # top-down smooth
        self.s3 = ConvBNReLU(lat_ch, lat_ch)    # after add/up from top
        self.s2 = ConvBNReLU(lat_ch, lat_ch)
        self.s1 = ConvBNReLU(lat_ch, lat_ch)

        # optional Re-Fuse at each level
        self.use_refuse = use_refuse
        if use_refuse:
            self.ref3 = ReFuseBlock(lat_ch)
            self.ref2 = ReFuseBlock(lat_ch)
            self.ref1 = ReFuseBlock(lat_ch)

        # final polish + classifier
        self.out = nn.Sequential(
            ConvBNReLU(lat_ch, lat_ch),
            nn.Conv2d(lat_ch, num_classes, 1, bias=True)
        )

    def forward(self, feats, ref_feats=None):
        """
        feats:    [p1,p2,p3,p4]  from fusion blocks (C-same)
        ref_feats (optional): tuple (u_list, s_list),
            where u_list=[u1..u4], s_list=[s1..s4] 与 feats 对齐，仅在 use_refuse=True 时用。
        """
        p1,p2,p3,p4 = feats
        # 统一通道
        l4 = self.l4(p4)
        l3 = self.l3(p3)
        l2 = self.l2(p2)
        l1 = self.l1(p1)

        # 顶层做 PPM，再自顶向下
        t4 = self.ppm(l4)                                # (B,C,H/16,W/16)
        t3 = self.s3(l3 + F.interpolate(t4, size=l3.shape[-2:], **up_kwargs))
        t2 = self.s2(l2 + F.interpolate(t3, size=l2.shape[-2:], **up_kwargs))
        t1 = self.s1(l1 + F.interpolate(t2, size=l1.shape[-2:], **up_kwargs))

        # 解码端再融合（可选）
        if self.use_refuse and (ref_feats is not None):
            u_list, s_list = ref_feats
            # 只在同分辨率上做（输入时已对齐）：
            t3 = self.ref3(t3, u_list[2], s_list[2])
            t2 = self.ref2(t2, u_list[1], s_list[1])
            t1 = self.ref1(t1, u_list[0], s_list[0])

        logits = self.out(t1)                             # 最终 logits 在 p1 分辨率
        return logits
