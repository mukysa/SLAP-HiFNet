# -*- coding: utf-8 -*-
"""辅助函数"""

# -- 系统模块 -- #
import time

# -- 第三方模块 -- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, f1_score, accuracy_score, confusion_matrix
from skimage.measure import label as sk_label
from skimage.morphology import binary_dilation, binary_erosion, square
import matplotlib.pyplot as plt

# -- 专有模块 -- #
from utils.custum_transform import edge_contour


# -- 损失函数 -- #

# 交叉熵损失函数
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, label_smoothing=0.1):
        super(CrossEntropyLoss, self).__init__()
        if weight is not None:
            weight = torch.from_numpy(np.array(weight)).float().cuda()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight, label_smoothing=label_smoothing)

    def forward(self, prediction, label):
        loss = self.ce_loss(prediction, label)

        return loss


# 边缘损失函数
class Edge_loss(nn.Module):

    def __init__(self, ignore_index=255):
        super(Edge_loss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, label):
        # h, w = label.size(1), label.size(2)
        pos_num = torch.sum(label == 1, dtype=torch.float)
        neg_num = torch.sum(label == 0, dtype=torch.float)
        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)

        weights = torch.Tensor([weight_neg, weight_pos])
        edge_loss = F.cross_entropy(prediction, label,
                                weights.cuda(), ignore_index=self.ignore_index)

        return edge_loss


# 距离加权交叉熵损失函数
class OrderedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(OrderedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction: torch.Tensor, label: torch.Tensor):

        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        loss = criterion(prediction, label)
        # calculate the hard predictions by using softmax followed by an argmax
        softmax = torch.nn.functional.softmax(prediction, dim=1)
        hard_prediction = torch.argmax(softmax, dim=1)
        # set the mask according to ignore index
        mask = label == self.ignore_index
        hard_prediction = hard_prediction[~mask]
        label = label[~mask]
        # calculate the absolute difference between target and prediction
        weights = torch.abs(hard_prediction-label) + 1
        # remove ignored index losses
        loss = loss[~mask]
        # if done normalization with weights the loss becomes of the order 1e-5
        # loss = (loss * weights)/weights.sum()
        loss = (loss * weights)
        loss = loss.mean()

        return loss
    

#  平均弱监督损失
class WindowWeaklyCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, window_size=1):
        super(WindowWeaklyCrossEntropyLoss, self).__init__()
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        self.ignore_index = ignore_index
        self.window_size = window_size

    def forward(self, prediction, label):
        pred_avg, label_avg = self.window_average(prediction, label)
        if pred_avg.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=prediction.device)
        return self.ce_loss(pred_avg, label_avg)

    def window_average(self, prediction, label):
        B, C, H, W = prediction.shape
        ws = self.window_size
        pred_unfold = prediction.unfold(2, ws, ws).unfold(3, ws, ws)
        label_unfold = label.unfold(1, ws, ws).unfold(2, ws, ws)
        B, C, num_h, num_w, _, _ = pred_unfold.shape
        pred_unfold = pred_unfold.contiguous().view(B, C, -1, ws * ws)
        label_unfold = label_unfold.contiguous().view(B, -1, ws * ws)
        valid_mask = label_unfold != self.ignore_index
        valid_counts = valid_mask.sum(dim=2)
        denom = valid_counts.unsqueeze(1).clamp(min=1)
        pred_sum = (pred_unfold * valid_mask.unsqueeze(1)).sum(dim=3)
        pred_avg = pred_sum / denom
        label_sum = (label_unfold * valid_mask).sum(dim=2)
        label_avg = (label_sum / valid_counts.clamp(min=1)).round().long()
        valid_windows = valid_counts > 0
        pred_avg = pred_avg.permute(0, 2, 1)[valid_windows]
        label_avg = label_avg[valid_windows]
        return pred_avg, label_avg


class SoftConsistencyLoss(nn.Module):
    def __init__(self, alpha=10.0, tau=0.5, ignore_index=255):
        super(SoftConsistencyLoss, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss()  # ← 这里用 logits 版本，自动内部 sigmoid

    def forward(self, pred_iw_logits: torch.Tensor, target_sic: torch.Tensor):
        # 注意这里的 pred_iw_logits 是 raw logits，不做 sigmoid
        # soft_label 是目标值（仍在 0~1 范围）
        soft_label = torch.sigmoid(self.alpha * (target_sic - self.tau))

        # 获取预测为类别1的logits
        if pred_iw_logits.shape[1] == 2:
            logits_iw = pred_iw_logits[:, 1, :, :]
        else:
            raise ValueError("Expected 2-class logits input for IW")

        # 避免掩膜
        valid_mask = (target_sic != self.ignore_index)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=target_sic.device)

        return self.bce(logits_iw[valid_mask], soft_label[valid_mask])


# 结构对比损失
def structure_contrastive_loss(features, label_mask, class_id=0, temperature=0.1, num_pos=3, num_neg=3, max_regions=20):
    """
    改进结构对比损失函数，GPU上进行正负采样，提升训练速度。

    Args:
        features: Tensor [B, C, H, W] — 中间层特征
        label_mask: Tensor [B, H, W] — 二分类标签（0=水，1=冰）
        class_id: int — 结构类别ID（0=水，1=冰）
        temperature: float — 对比温度
        num_pos: int — 每个 anchor 的正例数量
        num_neg: int — 每个 anchor 的负例数量
        max_regions: int — 每图最大采样区域数
    Returns:
        Scalar loss
    """
    B, C, H, W = features.shape
    loss = 0.0
    total = 0

    for b in range(B):
        feat = features[b].permute(1, 2, 0)  # [H, W, C]
        feat = F.normalize(feat, dim=-1)

        pred = label_mask[b].cpu().numpy()
        structure_mask = (pred == class_id).astype('uint8')
        connected = sk_label(structure_mask, connectivity=1)

        region_ids = list(set(connected.flatten()))
        region_coords = {}
        for rid in region_ids:
            if rid == 0: continue
            coords = torch.from_numpy(np.array((connected == rid).nonzero())).long()
            if len(coords) >= num_pos + 1:
                region_coords[rid] = coords

        region_ids = list(region_coords.keys())
        for rid in region_ids:
            coords = region_coords[rid]
            anchor_ids = torch.randperm(len(coords))[:num_pos]

            for a in anchor_ids:
                anchor_coord = coords[a]
                y, x = anchor_coord.tolist()
                if y >= H or x >= W: continue  # 越界防护
                anchor_feat = feat[y, x]

                # 正样本对
                pos_pool = coords[torch.randperm(len(coords))[:num_pos]]
                for pos in pos_pool:
                    py, px = pos.tolist()
                    if py >= H or px >= W: continue
                    pos_feat = feat[py, px]
                    sim = torch.dot(anchor_feat, pos_feat) / temperature
                    loss += -F.logsigmoid(sim)
                    total += 1

                # 负样本对
                neg_rids = [r for r in region_ids if r != rid]
                neg_sampled = 0
                while neg_sampled < num_neg and neg_rids:
                    neg_rid = neg_rids.pop(torch.randint(len(neg_rids), (1,)).item())
                    neg_coords = region_coords[neg_rid]
                    neg = neg_coords[torch.randint(len(neg_coords), (1,))]
                    ny, nx = neg.tolist()
                    if ny >= H or nx >= W: continue
                    neg_feat = feat[ny, nx]
                    sim = torch.dot(anchor_feat, neg_feat) / temperature
                    loss += -F.logsigmoid(-sim)
                    total += 1
                    neg_sampled += 1

    if total == 0:
        return torch.tensor(0.0, requires_grad=True, device=features.device)
    return loss / total


def _make_edge_band(labels: torch.Tensor,
                    radius: int = 1,
                    band_width: int = 3,
                    ignore_index: int = 255) -> torch.Tensor:
    """
    在线从标签生成边界带（1=边界/缓冲区，0=非边界）。
    labels: (B,H,W) int64
    返回: (B,1,H,W) float {0,1}
    """
    assert labels.dim() == 3, "labels should be (B,H,W)"
    B, H, W = labels.shape
    dev = labels.device

    # 有效区域
    valid = (labels != ignore_index)

    # 用 max/min pooling 做“形态学梯度”：邻域内类别不一致 => 边界
    l = labels.clone()
    l[~valid] = -1  # 忽略区标为 -1，避免误判
    lf = l.to(torch.float32).unsqueeze(1)  # (B,1,H,W)

    k = 2 * radius + 1
    pad = radius

    max_nb = F.max_pool2d(lf, kernel_size=k, stride=1, padding=pad)
    min_nb = -F.max_pool2d(-lf, kernel_size=k, stride=1, padding=pad)

    boundary = (max_nb != min_nb).to(torch.float32)  # (B,1,H,W)
    # 忽略像素一律置 0
    boundary = boundary * valid.unsqueeze(1).to(torch.float32)

    # 扩张得到“边界带”
    if band_width > 1:
        dil_k = 2 * (band_width // 2) + 1
        boundary = F.max_pool2d(boundary, kernel_size=dil_k, stride=1, padding=band_width // 2)

    return boundary.clamp(0, 1)


def edge_weighted_ce_loss(logits: torch.Tensor,
                          labels: torch.Tensor,
                          ignore_index: int = 255,
                          radius: int = 1,
                          band_width: int = 3,
                          edge_weight: float = 2.0,
                          base_weight: float = 1.0) -> torch.Tensor:
    """
    边界带加权的 Cross Entropy。
    logits : (B,C,H,W)
    labels : (B,H,W) int64
    返回   : 标量 loss
    """
    assert logits.dim() == 4 and labels.dim() == 3
    B, C, H, W = logits.shape
    assert labels.shape == (B, H, W)

    # per-pixel CE（不做平均）
    ce = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction='none')  # (B,H,W)

    # 边界带权重
    edge_band = _make_edge_band(labels, radius=radius, band_width=band_width, ignore_index=ignore_index)  # (B,1,H,W)
    weight = base_weight + (edge_weight - base_weight) * edge_band  # (B,1,H,W)
    weight = weight.squeeze(1)  # (B,H,W)

    valid = (labels != ignore_index).to(ce.dtype)
    ce = ce * weight * valid

    denom = (weight * valid).sum().clamp(min=1.0)
    return ce.sum() / denom


@torch.no_grad()
def _sic_expect_and_conf(logits_sic: torch.Tensor, nclass: int, detach: bool = True):
    """SIC logits -> 期望浓度[0,1] 与最大置信度"""
    x = logits_sic.detach() if detach else logits_sic
    prob = F.softmax(x, dim=1)                    # [B,C,H,W]
    denom = max(1, nclass - 1)
    vals = torch.arange(nclass, dtype=torch.float32, device=prob.device) / denom
    expect = (prob * vals.view(1, -1, 1, 1)).sum(dim=1)  # [B,H,W]
    maxp = prob.max(dim=1).values                        # [B,H,W]
    return expect, maxp

def conflict_edge_ce_loss(
    logits_iw: torch.Tensor,   # (B,2,H,W)
    labels_iw: torch.Tensor,   # (B,H,W)
    logits_sic: torch.Tensor,  # (B,C,H,W) 或 None
    options: dict                  # 直接传 stage2 的 options
):
    """
    在 IW 边界 & 与 SIC 冲突的像素上加大权重；带阶段内 warm-up。
    开关：opt['use_conflict_edge_loss']；关掉时回退为普通 CE。
    """
    ignore_index = int(options.get('ignore_index', 255))

    # 关 / 无 SIC -> 普通 CE
    if not options.get('use_conflict_edge_loss', False) or (logits_sic is None):
        return F.cross_entropy(logits_iw, labels_iw, ignore_index=ignore_index)

    device = logits_iw.device
    B, _, H, W = logits_iw.shape

    # --- 读配置 ---
    nclass          = int(options.get('nclass', 11))
    tau_low         = float(options.get('iw_conf_low', 0.10))
    tau_high        = float(options.get('iw_conf_high', 0.60))
    sic_conf_min    = float(options.get('iw_sic_conf_min', 0.60))
    edge_band       = int(options.get('iw_edge_band', 3))
    w_edge          = float(options.get('iw_w_edge', 0.5))
    w_conflict      = float(options.get('iw_w_conflict', 1.0))
    w_conflict_edge = float(options.get('iw_w_conflict_edge', 2.0))
    weight_clamp    = float(options.get('iw_weight_clamp', 4.0))
    detach_sic      = bool(options.get('iw_detach_sic', True))

    # 阶段内 warm-up（按 local epoch）
    cur_ep  = int(options.get('cur_epoch', 1))
    warm_ep = max(1, int(options.get('iw_conf_warmup_epochs', 5)))
    ramp    = min(1.0, max(0.0, cur_ep / float(warm_ep)))

    # --- 尺度对齐：把 SIC logits 对齐到 IW 分辨率 ---
    if logits_sic.shape[-2:] != (H, W):
        logits_sic = F.interpolate(logits_sic, size=(H, W), mode='bilinear', align_corners=False)

    # --- 从 SIC 得到“期望浓度”和置信度 ---
    p_sic = F.softmax(logits_sic.detach() if detach_sic else logits_sic, dim=1)   # (B,C,H,W)
    # 0..1 的浓度刻度
    label_values = torch.linspace(0.0, 1.0, steps=p_sic.shape[1], device=device).view(1, -1, 1, 1)
    sic_expect = (p_sic * label_values).sum(dim=1)                  # (B,H,W), ∈[0,1]
    sic_maxp   = p_sic.max(dim=1).values                            # (B,H,W)
    sic_conf_mask = (sic_maxp > sic_conf_min).float().unsqueeze(1)  # (B,1,H,W)

    # --- 冲突掩膜（两侧）：Ice↔Water 与 SIC 期望矛盾 ---
    lab = labels_iw
    m_conflict = (
        ((lab == 1) & (sic_expect < tau_low))  |   # IW=冰，但 SIC 很低 → 开阔水/水道嫌疑
        ((lab == 0) & (sic_expect > tau_high))     # IW=水，但 SIC 很高 → 薄冰/碎冰嫌疑
    ).float().unsqueeze(1).to(device)
    m_conflict = m_conflict * sic_conf_mask        # 只在 SIC 置信较高处生效

    # --- IW 边界带（用你现有的 edge_contour）---
    # 注意：edge_contour 接受 (B,H,W) 或 (H,W)，返回 0/1；若是 CPU 返回，记得搬回 device
    m_edge = edge_contour(lab, band=edge_band)
    if isinstance(m_edge, torch.Tensor):
        m_edge = m_edge.to(device)
    else:
        m_edge = torch.as_tensor(m_edge, device=device)
    m_edge = m_edge.unsqueeze(1).float()           # (B,1,H,W)

    # --- 合成权重（边界常开；冲突与“冲突∧边界”走 warm-up）---
    weight = 1.0 \
           + w_edge * m_edge \
           + ramp * (w_conflict * m_conflict + w_conflict_edge * (m_conflict * m_edge))  # 叠加项

    # 可选再乘 IW 置信（若你开启了 confidence_ce）
    if options.get('use_confidence_ce', False):
        conf_map = compute_confidence_map(logits_iw, method=options.get('conf_method', 'entropy'), return_raw=True)  # (B,1,H,W)
        weight = weight * conf_map.to(device)

    weight = torch.clamp(weight, max=weight_clamp)

    # --- 像素级 CE（忽略无效像素），做加权平均 ---
    pixel_ce = F.cross_entropy(logits_iw, lab, ignore_index=ignore_index, reduction='none').unsqueeze(1)  # (B,1,H,W)
    valid    = (lab != ignore_index).float().unsqueeze(1)
    loss = (weight * pixel_ce * valid).sum() / valid.sum().clamp(min=1.0)
    return loss


def compute_confidence_map(logits, method='max', threshold=0.7, return_raw=False):
    """
    logits: shape [B, C, H, W]
    method: 'max' or 'entropy'
    return_raw: 是否返回原始 confidence map，否则返回 soft mask（0/1）
    """
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        if method == 'max':
            conf = probs.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        elif method == 'entropy':
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  # [B, 1, H, W]
            conf = 1 - entropy / np.log(probs.shape[1])  # normalize to [0, 1]
        else:
            raise ValueError(f"Unsupported method: {method}")

        if return_raw:
            return conf  # [B, 1, H, W]
        else:
            return (conf > threshold).float()


# -- 评估指标 -- #

# R2·指标
def r2_metric(true, pred):
    """
    Calculate the r2 metric.
    R2计算

    Parameters
    ----------
    true : 
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.

    Returns
    -------
    r2 : float
        The calculated r2 score.
        
    """
    r2 = r2_score(y_true=true, y_pred=pred)
    
    return r2


# f1 score
def f1_metric(true, pred):
    """
    Calculate the weighted f1 metric.
    F1-score计算

    Parameters
    ----------
    true : 
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.

    Returns
    -------
    f1 : float
        The calculated f1 score.
        
    """
    f1 = f1_score(y_true=true, y_pred=pred, average='weighted')
    
    return f1


def _f1_group(y_true: np.ndarray, y_pred: np.ndarray, pos_vals):
    """
    把 pos_vals（集合/列表）视作“正类”，其他归为“负类”，计算二分类 F1。
    """
    pos_vals = set(pos_vals)
    t_bin = np.isin(y_true, list(pos_vals)).astype(np.uint8)
    p_bin = np.isin(y_pred, list(pos_vals)).astype(np.uint8)
    return f1_metric(t_bin, p_bin)


# accuracy
def accuracy_metric(true, pred):
    """
    Calculate the overall accuracy metric.

    Parameters
    ----------
    true : 
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.

    Returns
    -------
    accuracy : float
        The calculated accuracy score.
    """
    accuracy = accuracy_score(y_true=true, y_pred=pred)
    return accuracy


# mIoU
def miou_metric(true, pred, n_classes):
    """
    Calculate the mean Intersection over Union (mIoU) metric.

    Parameters
    ----------
    true : 
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.
    n_classes : int
        Number of classes in the classification task.

    Returns
    -------
    miou : float
        The calculated mIoU score.
    """
    cm = confusion_matrix(true, pred, labels=range(n_classes))
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou_per_class = intersection / union
    miou = np.nanmean(iou_per_class)
    return miou, iou_per_class


def boundary_f1(gt_bin: np.ndarray, pred_bin: np.ndarray, tol: int = 1) -> float:
    """
    Boundary-F1 with tolerance (像素容差 tol).
    计算：
      B = mask XOR erode(mask)
      P = |B_pred ∧ dilate(B_gt, tol)| / |B_pred|
      R = |B_gt  ∧ dilate(B_pred, tol)| / |B_gt|
      BF1 = 2PR/(P+R)
    注意：本实现不传 border_value，默认边界外为 False，兼容老版 skimage。
    """
    assert gt_bin.shape == pred_bin.shape
    # 保证布尔类型
    gt_bin   = gt_bin.astype(bool, copy=False)
    pred_bin = pred_bin.astype(bool, copy=False)

    se3 = square(3)
    # 旧版 skimage 接口是 (image, footprint/selem, out=None)，不接受 border_value
    b_gt   = np.logical_xor(gt_bin,  binary_erosion(gt_bin,  se3))
    b_pred = np.logical_xor(pred_bin, binary_erosion(pred_bin, se3))

    # 容差核
    tol = max(int(tol), 1)
    sed = square(2 * tol + 1)
    b_gt_dil   = binary_dilation(b_gt,   sed)
    b_pred_dil = binary_dilation(b_pred, sed)

    tp_p = (b_pred & b_gt_dil).sum()
    tp_r = (b_gt   & b_pred_dil).sum()
    p_den = max(int(b_pred.sum()), 1)
    r_den = max(int(b_gt.sum()),   1)

    precision = tp_p / p_den
    recall    = tp_r / r_den
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def slice_middle_classes(y_true: np.ndarray, y_pred: np.ndarray, nclass: int, lo: int = 1, hi: int = 8):
    """
    仅取 SIC 的中间类别 [lo, hi]（默认 1..8 ≈ 10%..80%）。
    过滤条件基于 y_true（更符合“在中间类上的表现”）。
    返回：y_true_mid, y_pred_mid（一维向量）
    """
    lo = max(lo, 0); hi = min(hi, nclass - 1)
    mid_mask = (y_true >= lo) & (y_true <= hi)
    return y_true[mid_mask], y_pred[mid_mask]


# -- 冻结主干 -- #
def freeze_backbone(net, freeze=True):
    for m in [net.transformer_backbone, net.cnn_backbone]:
        for p in m.parameters():
            p.requires_grad = not freeze


# -- 计算loss用时 -- #
def test_loss_timing(batch_size=4, height=1024, width=1024, num_classes=5, window_sizes=[2, 4, 8, 16, 32, 64, 128]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    time_official, time_new = [], []
    loss_official_list, loss_new_list = [], []

    for ws in window_sizes:
        print(f"Testing window size: {ws}")

        # Create input
        prediction = torch.randn(batch_size, num_classes, height, width, device=device)
        label = torch.randint(0, num_classes, (batch_size, height, width), device=device)
        ignore_mask = torch.rand(batch_size, height, width, device=device) < 0.1
        label[ignore_mask] = 255
        weight = torch.tensor([1.0] * num_classes, dtype=torch.float32, device=device)

        # Official
        start = time.time()
        official_loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        loss_official = official_loss_fn(
            prediction.view(batch_size, num_classes, -1).transpose(1, 2).reshape(-1, num_classes),
            label.view(-1)
        )
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_official.append(time.time() - start)
        loss_official_list.append(loss_official.item())
        print("Official done.")

        # New
        start = time.time()
        loss_fn_new = WindowWeaklyCrossEntropyLoss(weight=weight, ignore_index=255, window_size=ws).to(device)
        loss_new = loss_fn_new(prediction, label)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_new.append(time.time() - start)
        loss_new_list.append(loss_new.item())
        print("New done.")

    return window_sizes, (time_official, time_new), (loss_official_list, loss_new_list)


def plot_results(window_sizes, time_data, loss_data):
    time_official, time_new = time_data
    loss_official, loss_new = loss_data

    # Time plot
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, time_official, marker='o', label='Official CE Loss')
    plt.plot(window_sizes, time_new, marker='s', label='Custom CE (New)')
    plt.xlabel('Window Size')
    plt.ylabel('Time (seconds)')
    plt.title('Loss Computation Time vs Window Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('timing_plot.png')

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, loss_official, marker='o', label='Official CE Loss')
    plt.plot(window_sizes, loss_new, marker='s', label='Custom CE (New)')
    plt.xlabel('Window Size')
    plt.ylabel('Loss Value')
    plt.title('Loss Value vs Window Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_plot.png')


if __name__ == "__main__":
    window_sizes, time_data, loss_data = test_loss_timing()
    plot_results(window_sizes, time_data, loss_data)