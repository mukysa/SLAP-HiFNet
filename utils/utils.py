# -*- coding: utf-8 -*-
"""工具小函数和数据设置"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import faiss


# 映射
# 按SIC映射
BINS_SIC = [0, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100, 101, 256]
CLASSES_SIC = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255]

# 颜色字典
COLOURS = {'red': '\033[0;31m',
           'black': '\033[0m',
           'green': '\033[0;32m',
           'orange': '\033[0;33m',
           'purple': '\033[0;35m',
           'blue': '\033[0;34m',
           'cyan': '\033[96m',
           'yellow': '\033[93m',  # 与 orange 通常共用 ANSI 值
           }

def colour_str(word, colour: str):
    """Function to colour strings."""
    return COLOURS[colour.lower()] + str(word) + COLOURS['black']


# 记录训练信息的md文件
def save_options_markdown(options_dict, save_path, title):
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        for k, v in sorted(options_dict.items()):
            f.write(f"- **{k}**: `{v}`\n")
        f.write("\n\n")
        
def _fmt_metric(name: str, val: float) -> str:
    """R2保留四位小数；所有F1类和mIoU转百分数两位；其他默认四位。"""
    import numpy as _np
    if val is None or (isinstance(val, float) and _np.isnan(val)):
        return "nan"
    name_up = name.upper()
    if name_up.startswith('R2'):                     # R2 与 R2_mid[...]
        return f"{val:.4f}"
    if 'F1' in name_up or name_up.startswith('BF1@') or name_up == 'MIOU':
        return f"{val*100:.2f}"
    return f"{val:.4f}"
