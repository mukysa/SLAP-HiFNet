# -*- coding: utf-8 -*-
# -- Built-in modules -- #
import os

# --Proprietary modules -- #

# -- 环境变量 -- #
os.environ['SASIC_DATA'] = '/ssdfs/datahome/u10109/yushi/SASIC'  # SASIC数据位置
os.environ['SASIC_MA_DATA'] = '/ssdfs/datahome/u10109/yushi/SASIC_MA'  # SASIC_S2数据位置
os.environ['ENV'] = '/share/home/u10109/data/yushi/CTNet'  # 本项目位置

# 高分辨率输入数据
SAR_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'sar_80_hh',
    'sar_80_hv'
]

# 低分辨率输入数据
AMSR_VARIABLES = [
    # -- AMSR2 channels -- #
    'amsr2_18_h', 'amsr2_18_v',
    'amsr2_36_h', 'amsr2_36_v',
    'amsr2_89_h', 'amsr2_89_v'
]

LABELS_ST1 = ['label_80_sic']  # 第一阶段标签变量名列表

LABELS_ST2 = ['label_80_sic', 'label_manual']  # 第二阶段
LABELS_ST2_SIC = ['label_80_sic']  # 第二阶段 SIC
LABELS_ST2_IW = ['label_manual']  # 第二阶段 IW

LABELS_ST3 = ['label_80_sic', 'label_manual']  # 第三阶段

# ——阶段 1：Sentinel-1 + AMSR2 + label_80_sic——
stage1_options = {
    'model_name': 'rs_4cat',
    'stage_name': 's1_pretrain',
    
    # -- 项目位置 -- #
    'path_to_train_data': os.environ['SASIC_DATA'],  # 数据集位置
    'path_to_test_val_data': os.environ['SASIC_DATA'],  # 数据集位置
    'path_to_env': os.environ['ENV'],  # 本项目位置
    'json_path': 'dataset/datalist.json',  # 数据集列表路径
    'test_json_path': 'dataset/testlist.json',  # 测试集列表路径
    'exclude_json_path': 'dataset/excludelist.json',  # 排除列表路径
    'npy_path': 'dataset',  # 存储极值的位置
    'log_path': 'log',  # 日志文件位置
    'model_path': 'model_results',  # 模型保存位置

    # --网络设置-- #
    # UNet 编码器（SAR）
    'unet_encoder': 'resnet50',           # 或 'resnet50' 等
    'unet_encoder_weights': 'imagenet',   # SAR 是 3 通道，可以用 'imagenet'。若不是3通道请设 None

    # Swin（AMSR2）
    'swin_backbone': 'swin_small',  # tiny, small
    'pretrained': True,

    # 融合与解码
    'fusion_channels': 128,
    
    # 输出
    'net_mode': 'sic_only', # 第一阶段为sic_only

    # --输入大小设置-- #
    'nclass': 11,
    'in_channels': 9,
    'sar_patch_size': 1024,
    'amsr_patch_size': 32,
    'sar_resolution': 80,
    
    'sar_in_channels': 3,
    'amsr_in_channels': 6,

    # - 输入变量设置
    'sar_variables': SAR_VARIABLES,
    'amsr_variables': AMSR_VARIABLES,
    'labels': LABELS_ST1,  # 标签变量名列表

    # - dataloader设置
    'epoch_len': 200,  # 每个 epoch 的样本数
    'batch_size': 12,
    'val_batch_size': 24,  # 验证集的 batch_size
    'num_workers': 6,  # 用于数据加载的线程数
    'imgaug': False,  # 是否使用 imgaug 数据增强
    'min_pixels_thd': 0.5,  # 最小有效像素比例
    'ignore_values': 101,  # 忽略的值(从此值开始，大于此值的量均为mask)
    'window_size': 1024,  # 窗口大小(用于验证)
    'stride': 1024,  # 步长（用于验证）
    'loader_upsampling': 'nearest',  # 上采样方法
    
    'test_window_size': 2048,  # 测试时使用的窗口大小
    'test_stride': 2048,  # 测试时使用的步长
    
    'split': 'year',  # year或random，用于分割SASIC数据集
    'num_val_scenes': 20,  # 从 train_list 中随机采样以用于验证的场景数0
    'train_years': ["2017", "2018", "2019", "2020","2021", "2022", "2023"],  # 训练集年份
    'val_years': ["2024"],  # 验证集年份

    # --训练设置-- #
    'epochs': 60,  # 训练轮数
    'mark_epoch': 45,  # 预估的收敛阶段
    
    'seed': 2025,  # 随机种子

    # - 优化器设置
    'optimizer': 'adam',  # 优化器类型
    # adam
    'adam_lr': 1e-5,
    'adam_momentum': 0.9,
    'adam_weight_decay': 0,
    # sgd
    'sgd_lr': 5e-4,
    'sgd_momentum': 0.9,
    'sgd_weight_decay': 1e-2,

    # - 学习率衰减设置
    'eta_min_scale': 0,  # 学习率最小值的缩放因子, eta_min = lr * eta_min_scale

    # - 学习率调度器
    'scheduler_type': 'wcos',
    # cosine
    'T_max': 60,  # cos学习率衰减周期
    # step
    'step_size': 15,  # step下降自适应学习率降低的周期数
    'gamma': 0.5,  # 学习率降低的因子
    # warmup
    'warmup_epochs': 3,  # 热身轮数
    'power': 2.0,

    # - 损失函数设置
    'loss_function': 'CE',  # 损失函数类型
    'ignore_index': 255,  # 忽略的标签值
}

# ——阶段 2：微调到 人工标签——
stage2_options = dict(stage1_options)
stage2_options.update({
    'stage_name': 'manual_finetune',
    
    'path_to_train_data': os.environ['SASIC_MA_DATA'],  # 数据集位置
    'path_to_test_val_data': os.environ['SASIC_MA_DATA'],  # 数据集位置
    'json_path': 'dataset/datalist_ma_cis.json',
    'test_json_path': 'dataset/datalist_ma_cis.json',

    # --网络设置-- #
    'use_iw_head': True,
    
    # 第二阶段可为all, sic_only 和 iw_only
    'net_mode': 'all', 

    # --输入大小设置-- #
    'sar_patch_size': 1024,
    'amsr_patch_size': 32,
    'sar_resolution': 80,
    
    'test_window_size': 2048,  # 测试时使用的窗口大小
    'test_stride': 512,  # 测试时使用的步长

    # - 输入变量设置
    'labels': LABELS_ST2,  # all

    # - dataloader设置
    'epoch_len': 200,  # 每个 epoch 的样本数
    'batch_size': 12,
    'min_pixels_thd': 0.25,  # 最小有效像素比例
    
    'fold_id': 0,  # 使用第几个fold

    # --训练设置-- #
    'epochs': 60,  # 训练轮数
    'mark_epoch': 45,  # 预估的收敛阶段

    # - 优化器设置
    'optimizer': 'adam',  # 优化器类型
    # adam
    'adam_lr': 1e-6,
    'adam_momentum': 0.9,
    'adam_weight_decay': 0,
    # sgd
    'sgd_lr': 5e-4,
    'sgd_momentum': 0.9,
    'sgd_weight_decay': 1e-2,

    # - 学习率衰减设置
    'eta_min_scale': 0,  # 学习率最小值的缩放因子, eta_min = lr * eta_min_scale

    # - 学习率调度器
    'scheduler_type': 'wcos',
    # cosine
    'T_max': 60,  # cos学习率衰减周期
    # step
    'step_size': 10,  # step下降自适应学习率降低的周期数
    'gamma': 0.5,  # 学习率降低的因子
    # warmup
    'warmup_epochs': 10,  # 热身轮数15
    'power': 2.0,

    # - 损失函数设置
    'loss_function': 'CE',  # 损失函数类型
    
    # 一致性损失函相关参数
    'use_consistency_loss': True,
    'consistency_lambda': 0.2,  # 可调，建议从 0.1–0.3 试验
    'cons_alpha': 10.0,
    'cons_tau': 0.1,
    
    # egde loss
    'use_edge_loss': True,
    'edge_lambda': 0.1,          # [0.05, 0.2] 可调
    'edge_radius': 1,            # 邻域半径；r=1 即 3x3
    'edge_band_width': 5,        # 边界带厚度（像素）
    'edge_weight': 2.0,          # 边界带内权重倍数
})


# ——阶段 3：在伪标签数据集上拓展训练——
stage3_options = dict(stage2_options)
stage3_options.update({
    'stage_name': 'pseudo_finetune',
    
    'path_to_train_data': os.environ['SASIC_DATA'],  # 数据集位置
    'path_to_test_val_data': os.environ['SASIC_MA_DATA'],  # 数据集位置
    'train_json_path': 'dataset/datalist.json',  # 数据集列表路径
    'test_json_path': 'dataset/datalist_ma_cis.json',

    # --网络设置-- #
    'use_iw_head': True,
    
    # 第二阶段可为all, sic_only 和 iw_only
    'net_mode': 'all', 

    # --输入大小设置-- #
    'sar_patch_size': 1024,
    'amsr_patch_size': 32,
    'sar_resolution': 80,
    
    'test_window_size': 2048,  # 测试时使用的窗口大小
    'test_stride': 512,  # 测试时使用的步长

    # - 输入变量设置
    'labels': LABELS_ST3,  # all

    # - dataloader设置
    'epoch_len': 200,  # 每个 epoch 的样本数
    'batch_size': 12,
    'min_pixels_thd': 0.25,  # 最小有效像素比例
    
    'train_years': ["2017", "2018", "2019", "2020","2021", "2022", "2023", "2024"],  # 训练集年份
    'val_years': [],  # 验证集年份

    # --训练设置-- #
    'epochs': 60,  # 训练轮数
    'mark_epoch': 45,  # 预估的收敛阶段

    # - 优化器设置
    'optimizer': 'adam',  # 优化器类型
    # adam
    'adam_lr': 1e-6,
    'adam_momentum': 0.9,
    'adam_weight_decay': 0,
    # sgd
    'sgd_lr': 5e-4,
    'sgd_momentum': 0.9,
    'sgd_weight_decay': 1e-2,

    # - 学习率衰减设置
    'eta_min_scale': 0,  # 学习率最小值的缩放因子, eta_min = lr * eta_min_scale

    # - 学习率调度器
    'scheduler_type': 'wcos',
    # cosine
    'T_max': 60,  # cos学习率衰减周期
    # step
    'step_size': 10,  # step下降自适应学习率降低的周期数
    'gamma': 0.5,  # 学习率降低的因子
    # warmup
    'warmup_epochs': 10,  # 热身轮数15
    'power': 2.0,

    # - 损失函数设置
    'loss_function': 'CE',  # 损失函数类型
    
    # 一致性损失函相关参数
    'use_consistency_loss': False,
    'consistency_lambda': 0.2,  # 可调，建议从 0.1–0.3 试验
    'cons_alpha': 5.0,
    'cons_tau': 0.1,
    
    # confidence ce loss
    'use_confidence_ce': False,
    'conf_method': 'entropy',  # or 'max'
    'iw_main_head_weight': 2.0,  # 主损失
    'iw_aux_head_weight': 1.0,  # 辅助头分割损失
})