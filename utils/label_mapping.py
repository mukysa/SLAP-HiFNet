# -*- coding: utf-8 -*-
"""dataloader的标签映射函数"""
import numpy as np

from utils.utils import BINS_SIC, CLASSES_SIC


def map_labels(options, labels):
    """
    映射标签到新的类别
    """
    new_labels = np.zeros_like (labels)
    bins = options['bins']
    new_classes = options['map_classes']
    for i in range(len(bins) - 1):
        mask = (labels >= bins[i]) & (labels < bins[i + 1])
        new_labels[mask] = new_classes[i]
    return new_labels
