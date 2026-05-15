# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def edge_contour(labels: torch.Tensor,
                 band: int = 3,
                 ignore_index: int = 255,
                 connectivity: int = 4) -> torch.Tensor:
    """
    从语义标签提取边界带掩膜。
    - labels: (B,H,W) 或 (H,W) 的整型标签（0..K-1），可含 ignore_index
    - band:   边界缓冲宽度（像素）。0 表示只取1像素等高线；>0 表示膨胀成带状
    - connectivity: 4 或 8 邻域
    返回: 与 labels 同形状的 bool 掩膜
    """
    if labels.dim() == 2:
        x = labels.unsqueeze(0)
        squeeze_back = True
    elif labels.dim() == 3:
        x = labels
        squeeze_back = False
    else:
        raise ValueError("labels must be (H,W) or (B,H,W)")

    x = x.long()
    valid = (x != ignore_index)

    # 与左右、上下邻居比较（只在有效像素上比较）
    dh = torch.zeros_like(x, dtype=torch.bool)
    dv = torch.zeros_like(x, dtype=torch.bool)

    dh[:, :, :-1] = (x[:, :, :-1] != x[:, :, 1:]) & valid[:, :, :-1] & valid[:, :, 1:]
    dv[:, :-1, :] = (x[:, :-1, :] != x[:, 1:, :]) & valid[:, :-1, :] & valid[:, 1:, :]

    edge = dh | dv

    if connectivity == 8:
        d1 = torch.zeros_like(x, dtype=torch.bool)
        d2 = torch.zeros_like(x, dtype=torch.bool)
        d1[:, :-1, :-1] = (x[:, :-1, :-1] != x[:, 1:, 1:]) & valid[:, :-1, :-1] & valid[:, 1:, 1:]
        d2[:, :-1,  1:] = (x[:, :-1,  1:] != x[:, 1:, :-1]) & valid[:, :-1,  1:] & valid[:, 1:, :-1]
        edge |= (d1 | d2)

    # 膨胀成边界带
    if band and band > 0:
        k = 2 * band + 1
        edge = F.max_pool2d(edge.float(), kernel_size=k, stride=1, padding=band) > 0.5
        edge = edge & valid  # 仅保留有效像素

    if squeeze_back:
        edge = edge.squeeze(0)

    return edge