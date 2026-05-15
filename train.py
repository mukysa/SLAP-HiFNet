# -*- coding: utf-8 -*-
# -- 内置模块 / Built-in modules -- #
import os
import gc

# -- 第三方模块 / Third-party modules -- #
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datetime import datetime
import logging
import importlib

# -- 项目自定义模块 / Project-specific modules -- #

# 种子设置 / Seed setup
from utils.seed_tool import seed_everything
# 工具函数 / Utility functions
from utils.utils import save_options_markdown
from utils.split_train_val import split_randomly, split_by_year, get_manual_dataset_lists, get_manual_val_lists # 划分训练验证的相关函数 / Functions for splitting training and validation sets
from utils.functions import (CrossEntropyLoss, OrderedCrossEntropyLoss, SoftConsistencyLoss, edge_weighted_ce_loss,
                             compute_confidence_map, r2_metric, f1_metric, miou_metric)
from utils.scheduler import WarmUpCosineAnnealingScheduler
# 用于训练和验证的 data loader / Data loaders for training and validation
from dataloaders.loaders import SarAmsrTrainDataset, SarAmsrPatchValDataset, SarAmsrSenceValTestDataset, get_variable_options


def build_model(options):
    model_name = options['model_name']
    
    # -- 单模态模型 / Single-mode models -- #
    if model_name == 'resnet_sar':
        from models.resnet_single_mode_dual_head import ResNet_DualHead_SAR
        model = ResNet_DualHead_SAR(options)
    elif model_name == 'resnet_amsr':
        from models.resnet_single_mode_dual_head import ResNet_DualHead_AMSR2
        model = ResNet_DualHead_AMSR2(options)
    elif model_name == 'swin_sar':
        from models.swinT_single_mode_dual_head import SwinT_DualHead_SAR
        model = SwinT_DualHead_SAR(options)
    elif model_name == 'swin_amsr':
        from models.swinT_single_mode_dual_head import SwinT_DualHead_AMSR2
        model = SwinT_DualHead_AMSR2(options)

    # -- 早期融合 / Early fusion -- #
    # CNN / 卷积神经网络
    elif model_name == 'resnet':
        from models.resnet_dual_head import ResNet_DualHead
        model = ResNet_DualHead(options)
    elif model_name == 'deeplabv3p':
        from models.deeplabv3_plus_dual_head import DeepLabV3Plus_DualHead
        model = DeepLabV3Plus_DualHead(options)
    elif model_name == 'pspnet':
        from models.pspnet_dual_head import PSPNet_DualHead
        model = PSPNet_DualHead(options)
    elif model_name == 'xnet':
        from models.xnet_dual_head import XNet_DualHead
        model = XNet_DualHead(options)
    elif model_name == 'manet':
        from models.manet_dual_head import MANet_DualHead
        model = MANet_DualHead(options)
    elif model_name == 'unet':
        from models.unet_dual_head import UNet_DualHead
        model = UNet_DualHead(options)
    elif model_name == 'unetpp':
        from models.unetpp_dual_head import UNetPP_DualHead
        model = UNetPP_DualHead(options)
    # Transformer / Transformer 网络
    elif model_name == 'cswin':
        from models.cswin_dual_head import CSWin_DualHead
        model = CSWin_DualHead(options)
    elif model_name == 'swin':
        from models.swinT_dual_head import SwinT_DualHead
        model = SwinT_DualHead(options)
    
    # -- 后期融合 / Late fusion -- #
    elif model_name == 'rr_p4cat':
        from models.rr_p4cat import RRp4cat
        model = RRp4cat(options)
    elif model_name == 'ss_p4cat':
        from models.ss_p4cat import SSp4cat
        model = SSp4cat(options)
    
    # -- 分层融合 / Hierarchical fusion -- #
    elif model_name == 'rr_4cat':
        from models.rr_4cat import RR4cat
        model = RR4cat(options)
    elif model_name == 'ss_4cat':  # 4组：SsSs、StSt、StSs、SsSt / Four variants: SsSs, StSt, StSs, and SsSt
        from models.ss_4cat import SS4cat
        model = SS4cat(options)
    
    # -- 本文方法 / Ours -- #    
    elif model_name == 'slapnet':
        from models.slapnet import SLAPNet
        model = SLAPNet(options)
        
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")

    return model


def load_options_by_model(model_name: str):
    """
    根据模型名从 options 模块动态导入配置。
    Dynamically import configuration options from the options module according to the model name.

    要求对应的 options 脚本中包含 stage1_options、stage2_options 和 stage3_options（如果需要）。
    The corresponding options script should contain stage1_options, stage2_options, and stage3_options if needed.
    """
    try:
        options_module = importlib.import_module(f"options.{model_name}")
    except ModuleNotFoundError:
        raise ImportError(f"Cannot find options module for model '{model_name}'. Expected: options/{model_name}.py")

    stage1 = getattr(options_module, "stage1_options", None)
    stage2 = getattr(options_module, "stage2_options", None)
    stage3 = getattr(options_module, "stage3_options", None)

    if stage1 is None or stage2 is None:
        raise ValueError(f"Missing stage1_options or stage2_options in options.{model_name}")
    
    return stage1, stage2, stage3


def train(net, loss_functions, optimizer, scheduler, dataloader, dataloader_val, train_options, epoch_offset=0):
    best_composite_score = 0  # 最佳的得分 / Best score
    last_best_composite_score = 0  # 收敛阶段的最佳得分 / Best score in the later convergence stage
    device = train_options['device']
    model_path = train_options['model_path']
    best_path = os.path.join(model_path, 'best_model')
    last_path = os.path.join(model_path, 'last_model')
    last_best_path = os.path.join(model_path, 'last_best_model')

    print(f"Training {train_options['model_name']}...")
    print()
    #  -- 训练 / Training -- #
    for epoch in range(train_options['epochs']):
        gc.collect()  # 释放内存 / Release memory
        train_loss_sum = torch.tensor([0.])  # 用于计算平均 train loss / Used to compute the average training loss
        val_loss_sum = torch.tensor([0.])  # 用于计算平均 val loss / Used to compute the average validation loss
        net.train()  # 将模型设置为训练模式 / Set the model to training mode
        # torch.autograd.set_detect_anomaly(True)  # 开启梯度异常检测 / Enable gradient anomaly detection

        # -- 训练一个 epoch / Train one epoch -- #
        print(f"Training ...")
        scaler = torch.cuda.amp.GradScaler()  # 初始化 GradScaler / Initialize GradScaler
        # torch.autograd.set_detect_anomaly(True)  # 开启梯度异常检测 / Enable gradient anomaly detection
        
        global_epoch = epoch_offset + epoch + 1  # 累加的 epoch / Accumulated epoch index
        train_tq = tqdm(iterable=dataloader, total=train_options['epoch_len'], # 准备 tqdm 进度条 / Prepare the tqdm progress bar
                        desc=f"Epoch {global_epoch}/{epoch_offset+train_options['epochs']}", mininterval=0.1, ncols=100)
        train_loss_epoch = 0  # 提取平均 loss，便于后续绘图查看下降趋势 / Track the average loss for plotting the decreasing trend
        postfix_info = {}  # 创建一个空字典用于存储后缀信息 / Create an empty dictionary for postfix information
        # -- 训练循环 / Training loop -- #
        cur_iters = epoch * train_options['epoch_len']  # 记录当前的迭代次数 / Record the current iteration count
        for it, (batch_hr, label_dict) in enumerate(train_tq):
            train_loss_list = []

            # - 转移到 device / Move tensors to device
            batch_hr = batch_hr.to(device, non_blocking=True)
            batch_sic = label_dict.get('label_80_sic', None)
            if batch_sic is not None: batch_sic = batch_sic.squeeze(1).to(device)
            batch_iw = label_dict.get('label_manual', None)
            if batch_iw is not None: batch_iw = batch_iw.squeeze(1).to(device)

            # - 混合精度训练（节省内存）/ Mixed-precision training to reduce memory usage
            with torch.cuda.amp.autocast():
                # - 前向传播 / Forward pass
                outputs_dict_train = net(batch_hr, mode=train_options['net_mode'])
                outputs_sic = outputs_dict_train.get('label_80_sic', None)
                outputs_iw = outputs_dict_train.get('label_manual', None)
                
                # >>> 主输出 logits 提取（放在这里，确保任何分支都能用到）/ Extract main logits here so all branches can access them
                iw_logits_main  = (outputs_iw[0]  if isinstance(outputs_iw,  (list, tuple)) else outputs_iw)  if outputs_iw  is not None else None
                sic_logits_main = (outputs_sic[0] if isinstance(outputs_sic, (list, tuple)) else outputs_sic) if outputs_sic is not None else None

                # - 计算损失 / Compute losses
                # SIC / 海冰密集度
                if outputs_sic is not None:
                    if train_options['loss_function'] == 'CE' or train_options['loss_function'] == 'OCE':
                        if isinstance(outputs_sic, (list, tuple)):
                            for i in range(len(outputs_sic)):
                                train_loss_list.append(loss_functions['ce_loss'](prediction=outputs_sic[i], label=batch_sic))
                        else:
                            train_loss_list.append(loss_functions['ce_loss'](prediction=outputs_sic, label=batch_sic))
                # IW（stage2, 3）/ Ice-water classification in stages 2 and 3
                if outputs_iw is not None:
                    if train_options.get('use_confidence_ce', False):
                        if isinstance(outputs_iw, (list, tuple)):
                            for i in range(len(outputs_iw)):
                                logits_iw = outputs_iw[i]
                                if i == 0:
                                    # 主头使用 confidence 加权 soft CE / Use confidence-weighted soft CE for the main head
                                    confidence_map = compute_confidence_map(logits_iw, method=train_options['conf_method'],return_raw=True)  # 返回 [B,1,H,W] / Return [B,1,H,W]
                                    pixel_loss = F.cross_entropy(logits_iw, batch_iw, ignore_index=train_options['ignore_index'],reduction='none').unsqueeze(1)  # [B,H,W] -> [B,1,H,W] / Convert [B,H,W] to [B,1,H,W]
                                    valid_mask = (batch_iw != train_options['ignore_index']).float().unsqueeze(1)
                                    weighted_loss = (pixel_loss * confidence_map * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
                                    train_loss_list.append(train_options['iw_main_head_weight'] * weighted_loss)
                                else:
                                    # 辅助头使用普通 CE / Use standard CE for auxiliary heads
                                    ce_loss = loss_functions['ce_loss'](prediction=logits_iw, label=batch_iw)
                                    train_loss_list.append(train_options['iw_aux_head_weight'] * ce_loss)
                        else:
                            # 单个 head 情况 / Single-head case
                            logits_iw = outputs_iw
                            confidence_map = compute_confidence_map(logits_iw, method=train_options['conf_method'], return_raw=True)
                            pixel_loss = F.cross_entropy(logits_iw, batch_iw, ignore_index=train_options['ignore_index'], reduction='none').unsqueeze(1)
                            valid_mask = (batch_iw != train_options['ignore_index']).float().unsqueeze(1)
                            weighted_loss = (pixel_loss * confidence_map * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
                            train_loss_list.append(weighted_loss)
                    else:
                        if train_options['loss_function'] == 'CE' or train_options['loss_function'] == 'OCE':
                            if isinstance(outputs_iw, (list, tuple)):
                                for i in range(len(outputs_iw)):
                                    train_loss_list.append(loss_functions['ce_loss'](prediction=outputs_iw[i], label=batch_iw))
                            else:
                                train_loss_list.append(loss_functions['ce_loss'](prediction=outputs_iw, label=batch_iw))
                    if train_options.get('use_edge_loss', False) and iw_logits_main is not None:
                        edge_loss_val = edge_weighted_ce_loss(
                            logits=iw_logits_main,           # (B,2,H,W) / IW logits shape
                            labels=batch_iw,                 # (B,H,W) / IW label shape
                            ignore_index=train_options['ignore_index'],
                            radius=train_options.get('edge_radius', 1),
                            band_width=train_options.get('edge_band_width', 3),
                            edge_weight=train_options.get('edge_weight', 2.0),
                            base_weight=1.0
                        )
                        train_loss_list.append(train_options.get('edge_lambda', 0.1) * edge_loss_val)
                
                # 任务间一致性损失 / Inter-task consistency loss
                if 'consistency_loss' in loss_functions and outputs_sic is not None and outputs_iw is not None:
                    # 注意：CE 任务中 outputs_sic 是 logits，要先转换为概率 / Note: outputs_sic contains logits for CE and should be converted to probabilities first
                    prob_sic = F.softmax(sic_logits_main, dim=1)  # 假设为 N 类 / Assume N classes
                    # 再将 label 映射回真实 [0, 1] 值 / Map class labels back to real-valued [0, 1] concentrations
                    # 举例：二分类 0-100（class_id）-> 浓度值 / 100.0 / Example: class IDs from 0 to 100 are converted back to concentration values divided by 100.0
                    label_values = torch.arange(train_options['nclass'], dtype=torch.float32, device=prob_sic.device) / max(1, (train_options['nclass'] - 1))
                    pred_sic_value = (prob_sic * label_values.view(1, -1, 1, 1)).sum(dim=1)
                    loss_consistency = loss_functions['consistency_loss'](iw_logits_main, pred_sic_value)
                    train_loss_list.append(train_options['consistency_lambda'] * loss_consistency)

            # 汇总多个输出头的 loss / Sum losses from multiple output heads
            loss_batch = sum(train_loss_list)  # 不使用 .mean() / Do not apply .mean()

            # - 重置上一轮的梯度 / Reset gradients from the previous iteration
            optimizer.zero_grad()
            # - 反向传播 / Backpropagation
            scaler.scale(loss_batch).backward()  # 使用 scaler 缩放梯度 / Scale gradients with the scaler

            # - 梯度裁剪 / Gradient clipping
            scaler.unscale_(optimizer)  # 在裁剪梯度之前还原缩放 / Unscale gradients before clipping

            # - 优化器更新 / Optimizer update
            scaler.step(optimizer)  # 使用 scaler 更新权重 / Update weights with the scaler
            scaler.update()  # 更新 scaler 的缩放因子 / Update the scaler factor
            
            # - 累计损失 / Accumulate loss
            train_loss_sum += loss_batch.detach().item()
            # - 计算平均损失 / Compute average loss
            train_loss_epoch = torch.true_divide(train_loss_sum, it + 1).detach().item()

            # 更新后缀信息 / Update postfix information
            postfix_info['mean_loss'] = f'{train_loss_epoch:.3f}'
            # 在 tqdm 进度条中显示后缀信息 / Display postfix information in the tqdm progress bar
            train_tq.set_postfix(postfix_info)
        
        # 在一个 epoch 的所有批次训练完成后更新学习率 / Update the learning rate after all batches in one epoch are trained
        cur_iters += train_options['epoch_len']
        if train_options['scheduler_type'] == 'wcos':
            scheduler.step(cur_iters)           # 以“迭代”为步长 / Step by iteration
        else:
            scheduler.step()                    # 以“epoch”为步长 / Step by epoch
        del train_loss_sum  # 释放内存（删除这条会导致奇怪的错误）/ Release memory; removing this line may cause unexpected errors

        # -- 验证循环 / Validation loop -- #
        val_loss_sum = 0.0  # 用于计算平均验证损失 / Used to compute the average validation loss
        outputs_sic_flat, labels_sic_flat = np.array([]), np.array([])
        outputs_iw_flat, labels_iw_flat = np.array([]), np.array([])

        net.eval()  # 将模型设置为评估模式 / Set the model to evaluation mode
        # - 循环验证集 / Iterate over the validation set
        print('Validating...')
        val_tq = tqdm(iterable=dataloader_val, total=len(train_options['validate_list']),  # 准备 tqdm 进度条 / Prepare the tqdm progress bar
                      desc=f"Epoch {global_epoch}/{epoch_offset+train_options['epochs']}", mininterval=0.1, ncols=100)
        for iv, batch in enumerate(val_tq):
            has_sic = False
            has_iw = False
            # 兼容 patch-window 与整幅影像格式 / Compatible with patch-window and full-scene formats
            if train_options['stage_name'] == 's1_pretrain':
                hr_scene, val_batch_label_sic, val_batch_mask, name = batch
                # 转移到设备 / Move tensors to device
                try:
                    hr_scene = hr_scene.to(device, non_blocking=True)
                    val_batch_label_sic = val_batch_label_sic.squeeze(1).to(device, non_blocking=True)  # 去掉多余的通道维度 / Remove the redundant channel dimension
                    val_batch_mask = val_batch_mask.squeeze(1).numpy()  # 去掉多余的通道维度 / Remove the redundant channel dimension
                except:
                    raise ValueError(f"The shape of val_batch_chart or val_batch_mask for scene '{name}' is problematic. Please move this scene to the exclusion list.")
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    # 前向传播 / Forward pass
                    outputs_dict_val = net(hr_scene, mode=train_options['net_mode'])
                    outputs_sic = outputs_dict_val.get('label_80_sic', None)
                    
                    # 计算验证损失 / Compute validation loss
                    if isinstance(outputs_sic, (list, tuple)):
                        outputs_sic = outputs_sic[0]  # 只使用主头输出 / Use only the main-head output

                    if train_options['loss_function'] == 'CE' or train_options['loss_function'] == 'OCE':
                        val_loss = loss_functions['ce_loss'](prediction=outputs_sic, label=val_batch_label_sic)

                    # 将输出和标签展平，用于计算 R2 分数 / Flatten outputs and labels to compute the R2 score
                    outputs_sic = torch.argmax(outputs_sic, dim=1).cpu().numpy()
                    val_batch_label_sic = val_batch_label_sic.cpu().numpy()

                    # 排除无效像素 / Exclude invalid pixels
                    valid_indices = ~val_batch_mask
                    outputs_sic_flat = np.append(outputs_sic_flat, outputs_sic[valid_indices])
                    labels_sic_flat = np.append(labels_sic_flat, val_batch_label_sic[valid_indices])
                    
                val_loss_sum += val_loss.item()
            else:
                # ----- 阶段二：整幅验证，含 pad 剪裁 / Stage 2: full-scene validation with padded-area cropping -----
                hr_windows, label_dict, mask, window_positions, scene_shape, pad_shape, sar, name = batch
                H, W = scene_shape
                ws = train_options['test_window_size']
                has_sic = 'label_80_sic' in label_dict
                has_iw  = 'label_manual' in label_dict
                val_loss_list = []

                # 初始化 logits 累加器和计数器 / Initialize logits accumulators and the count matrix
                if has_sic: logits_sic_sum = torch.zeros((1, train_options['nclass'], *pad_shape), dtype=torch.float32).to(device)
                if has_iw: logits_iw_sum = torch.zeros((1, 2, *pad_shape), dtype=torch.float32).to(device)
                count_matrix = torch.zeros(pad_shape, dtype=torch.int32)

                # 推理每个窗口 / Run inference for each window
                with torch.no_grad():
                    for idx, (hr_window, (i, j)) in enumerate(zip(hr_windows, window_positions)):
                        hr_window = hr_window.unsqueeze(0).to(device, non_blocking=True)

                        with torch.cuda.amp.autocast():
                            outputs_dict_val = net(hr_window, mode=train_options['net_mode'])
                            outputs_sic = outputs_dict_val.get('label_80_sic', None)
                            outputs_iw = outputs_dict_val.get('label_manual', None)
                            if isinstance(outputs_sic, (list, tuple)): outputs_sic = outputs_sic[0]
                            if isinstance(outputs_iw, (list, tuple)): outputs_iw = outputs_iw[0]

                        if has_sic: logits_sic_sum[:, :, i:i+ws, j:j+ws] += outputs_sic.detach()
                        if has_iw: logits_iw_sum[:, :, i:i+ws, j:j+ws] += outputs_iw.detach()
                        count_matrix[i:i+ws, j:j+ws] += 1
                        del hr_window, outputs_dict_val, outputs_sic, outputs_iw
                
                count_matrix = torch.clamp(count_matrix, min=1).to(device)
                # 掩膜处理 / Mask processing
                if isinstance(mask, torch.Tensor): mask = mask.numpy()
                mask = mask[:H, :W]
                valid_indices = ~mask
                
                if has_sic:
                    # 取平均 logits，并裁剪到有效区域 / Average logits and crop to the valid area
                    logits_sic_avg = logits_sic_sum / count_matrix.unsqueeze(0).unsqueeze(0)
                    logits_sic_avg = logits_sic_avg[:, :, :H, :W]
                    # 分类结果用于计算精度 / Classification results for accuracy evaluation
                    full_output_sic = torch.argmax(logits_sic_avg, dim=1).squeeze(0).cpu().numpy()
                    # 标签和掩膜处理 / Label and mask processing
                    gt_sic = label_dict['label_80_sic'][:H, :W]
                    if isinstance(gt_sic, torch.Tensor): gt_sic = gt_sic.numpy()
                    # 计算验证损失 / Compute validation loss
                    labels_sic_tensor = torch.from_numpy(gt_sic).unsqueeze(0).long().to(device)
                    loss_sic = loss_functions['ce_loss'](logits_sic_avg, labels_sic_tensor)
                    val_loss_list.append(loss_sic)
                    # 保存精度评估指标 / Store values for accuracy metrics
                    outputs_sic_flat = np.append(outputs_sic_flat, full_output_sic[valid_indices])
                    labels_sic_flat = np.append(labels_sic_flat, gt_sic[valid_indices])
                
                if has_iw:
                    # 取平均 logits，并裁剪到有效区域 / Average logits and crop to the valid area
                    logits_iw_avg = logits_iw_sum / count_matrix.unsqueeze(0).unsqueeze(0)
                    logits_iw_avg = logits_iw_avg[:, :, :H, :W]
                    # 分类结果用于计算精度 / Classification results for accuracy evaluation
                    full_output_iw = torch.argmax(logits_iw_avg, dim=1).squeeze(0).cpu().numpy()
                    # 标签和掩膜处理 / Label and mask processing
                    gt_iw = label_dict['label_manual'][:H, :W]
                    if isinstance(gt_iw, torch.Tensor): gt_iw = gt_iw.numpy()
                    # 计算验证损失 / Compute validation loss
                    labels_iw_tensor = torch.from_numpy(gt_iw).unsqueeze(0).long().to(device)
                    loss_iw = loss_functions['ce_loss'](logits_iw_avg, labels_iw_tensor)
                    val_loss_list.append(loss_iw)
                    # 保存精度评估指标 / Store values for accuracy metrics
                    outputs_iw_flat = np.append(outputs_iw_flat, full_output_iw[valid_indices])
                    labels_iw_flat = np.append(labels_iw_flat, gt_iw[valid_indices])
                
                # 计算总验证损失 / Compute the total validation loss
                val_loss = sum(val_loss_list)  # 不使用 .mean() / Do not apply .mean()
                val_loss_sum += val_loss.item()
                
                if has_sic:
                    del logits_sic_sum, logits_sic_avg, labels_sic_tensor
                if has_iw:
                    del logits_iw_sum, logits_iw_avg, labels_iw_tensor
                del hr_windows, label_dict, mask, window_positions, scene_shape, pad_shape, sar

        # - 计算 val loss / Compute validation loss
        val_loss_epoch = torch.true_divide(val_loss_sum, iv + 1).detach().item()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # - 计算指标（第二阶段）/ Compute metrics for the second-stage validation
        if train_options['stage_name'] == 's1_pretrain' or has_sic:
            r2_score = r2_metric(labels_sic_flat, outputs_sic_flat)
            f1_score_sic = f1_metric(labels_sic_flat, outputs_sic_flat)
        else:
            r2_score, f1_score_sic = 0.0, 0.0
        if has_iw:
            f1_score_iw = f1_metric(labels_iw_flat, outputs_iw_flat)
            miou_iw, _ = miou_metric(labels_iw_flat, outputs_iw_flat, n_classes=2)
        else:
            f1_score_iw, miou_iw = 0.0, 0.0

        # - 计算综合分数 / Compute the composite score
        if train_options['stage_name'] == 's1_pretrain':
            composite_score = 0.3 * r2_score + 0.7 * f1_score_sic
        elif has_sic and has_iw:
            composite_score = 0.15 * r2_score + 0.35 * f1_score_sic + 0.15 * f1_score_iw + 0.35 * miou_iw
        elif has_sic:
            composite_score = 0.3 * r2_score + 0.7 * f1_score_sic
        elif has_iw:
            composite_score = 0.3 * f1_score_iw + 0.7 * miou_iw
        else:
            composite_score = 0.0

        # - 打印 / Print results
        print(f"Final batch loss: {train_loss_epoch:.3f}")
        print(f"Validation loss: {val_loss_epoch:.3f}")
        print(f"Epoch {global_epoch} score:")
        print(f"SIC R2 score: {r2_score:.3f}")
        print(f"SIC F1 score: {f1_score_sic:.3f}")
        print(f"IW  F1 score: {f1_score_iw:.3f}")
        print(f"IW  mIoU score: {miou_iw:.3f}")
        print(f"Composite score: {composite_score:.3f}")
        print("")
        
        # -- log 记录 / Logging -- #
        logging.info(f'{global_epoch}, {train_loss_epoch:.3f}, {val_loss_epoch:.3f}, {r2_score:.3f}, {f1_score_sic:.3f}, {f1_score_iw:.3f}, {miou_iw:.3f}, {composite_score:.3f}')

        # 绘图 TODO / Plotting TODO

        # -- 保存模型 / Save models -- #
        # 如果分数高于历史最佳，则保存为 best_model / If the score exceeds the previous best, save it as best_model
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            torch.save(obj={'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch+1},
                       f=best_path)
        # 存储最后 epoch 的模型（每 5 个 epoch）/ Save the latest model every 5 epochs
        if (epoch+1) % 5 == 0:
            torch.save(obj={'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch+1},
                        f=last_path)
        # 获取后半段最佳模型 / Save the best model in the later training stage
        if (epoch+1) >= train_options['mark_epoch'] and composite_score > last_best_composite_score:
            last_best_composite_score = composite_score
            torch.save(obj={'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch+1},
                        f=last_best_path)
    return last_best_path  # 或 last_path / or last_path


def run_stage(train_options, resume_ckpt=None, epoch_offset=0):
    print(f"Training stage: {train_options['stage_name']}")

    # ------- 将 stage-specific opts 送入 get_variable_options / Pass stage-specific opts to get_variable_options -------
    train_options = get_variable_options(train_options)
    print('Loading of training options complete.')
    
     # -- 加载训练列表 / Load training lists -- #
    json_file_path = os.path.join(train_options['path_to_env'], train_options['json_path'])
    if train_options['stage_name']=='s1_pretrain':  # stage1 / 第一阶段
        if train_options['split'] == 'year':  # 按年份划分训练验证集 / Split training and validation sets by year
            train_options['train_list'], train_options['validate_list'] = split_by_year(json_file_path, train_years=train_options['train_years'],
                                                                            val_years=train_options['val_years'])
        elif train_options['split'] == 'random':  # 按固定验证场景数随机划分训练验证集 / Randomly split training and validation sets using a fixed number of validation scenes
            train_options['train_list'], train_options['validate_list'] = split_randomly(json_file_path, val_count=train_options['num_val_scenes'],
                                                                                        exclude_files=train_options['exclude_json_path'])
        else:
            raise ValueError(f"Unsupported split method: {train_options['split']}")
    elif train_options['stage_name']=='manual_finetune':  # stage2 / 第二阶段
       train_options['train_list'], train_options['validate_list'], train_options['test_list'] = get_manual_dataset_lists(json_file_path, fold_id=train_options['fold_id'])
    elif train_options['stage_name'] == 'pseudo_training':  # stage3 / 第三阶段
        train_json_path = os.path.join(train_options['path_to_env'], train_options['train_json_path'])
        test_json_path = os.path.join(train_options['path_to_env'], train_options['test_json_path'])
        # 实际使训练集为 stage1 的 train + val / In practice, use stage1 train + val as the training set
        train_options['train_list'], _ = split_by_year(train_json_path, train_years=train_options['train_years'], val_years=[])
        # 实际使验证集为 stage2 的 train + val / In practice, use stage2 train + val as the validation set
        train_options['validate_list'] = get_manual_val_lists(test_json_path)
    else:
        raise ValueError(f"Unknown stage_name: {train_options['stage_name']}")
    print('Training validation set initialisation complete.')
    
    # -- dataloader 设置 / Dataloader setup
    # 训练数据及加载器（鉴于本数据集的获取特性，无法固定 dataloader 的 seed）/ Training dataset and loader; the dataloader seed cannot be fixed due to how this dataset is sampled
    dataset_train = SarAmsrTrainDataset(train_options)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=None, shuffle=True, num_workers=train_options['num_workers'], pin_memory=True)
    if train_options['stage_name']=='s1_pretrain':
        dataset_val   = SarAmsrPatchValDataset(train_options, train_options['validate_list'])
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=None, shuffle=False, num_workers=train_options['num_workers'], pin_memory=True)
    else:
        dataset_val   = SarAmsrSenceValTestDataset(train_options, train_options['validate_list'])
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=None, shuffle=False, num_workers=0, pin_memory=False)
    print('Dataloader initialisation complete.')
    
    # -- 模型初始化 / Model initialization -- #
    net = build_model(train_options)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net = net.to(train_options['device'])
    print('Model initialisation complete.')
    
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=train_options['device'])
        net.load_state_dict(ckpt['model_state_dict'], strict=False)  # strict=False 兼容新标签头 / strict=False keeps compatibility with newly added label heads
        print(f"[{train_options['stage_name']}] loaded ckpt -> {resume_ckpt}")
    
    # -- 优化器设置 / Optimizer setup -- #
    if train_options['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['adam_lr'], betas=(train_options['adam_momentum'], 0.999),
                                    weight_decay=train_options['adam_weight_decay'])
        train_options['eta_min'] = train_options['adam_lr'] * train_options['eta_min_scale']
    else:
        raise ValueError(f"Unsupported optimizer type: {train_options['optimizer']}")

    # -- 初始化学习率调度器 / Initialize the learning-rate scheduler -- #
    if train_options['scheduler_type'] == 'wcos':
        scheduler = WarmUpCosineAnnealingScheduler(
            optimizer=optimizer,
            base_lr=train_options['adam_lr'] if train_options['optimizer'] == 'adam' else train_options['sgd_lr'],
            min_lr=train_options['eta_min'],
            max_iters=train_options['epochs'] * train_options['epoch_len'],
            warmup_iters=train_options['warmup_epochs'] * train_options['epoch_len'],
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {train_options['scheduler_type']}")
    
    # -- 损失函数 / Loss functions -- #
    loss_functions = {}
    # 主任务损失函数 / Main-task loss function
    if train_options['loss_function'] == 'CE':
        loss_functions['ce_loss'] = CrossEntropyLoss(weight=None, ignore_index=train_options['ignore_index'])
    elif train_options['loss_function'] == 'OCE':
        loss_functions['ce_loss'] = OrderedCrossEntropyLoss(ignore_index=train_options['ignore_index'])
    else:
        raise ValueError(f"Unsupported loss_function type: {train_options['loss_function']}")
    # 辅助损失函数（soft 一致性函数）/ Auxiliary loss function: soft consistency loss
    if train_options.get('use_consistency_loss', False):
        loss_functions['consistency_loss'] = SoftConsistencyLoss(alpha=train_options.get('cons_alpha', 10.0),
                                                                 tau=train_options.get('cons_tau', 0.5),
                                                                 ignore_index=train_options['ignore_index'])
    
    print('Loss function setup complete.')
    
    # -- 训练 / Training
    model_path = train(net, loss_functions, optimizer, scheduler, dataloader_train, dataloader_val, train_options, epoch_offset)
    
    print(f"Training stage: {train_options['stage_name']} Complete!")
    print()
    
    return model_path, train_options['epochs']


def setup_training_stage(options_list, train_mode, current_time):
    """
    接收多个阶段的 options，统一设置 save_name、log_name、model_path 并创建目录。
    Receive options from multiple stages, set save_name, log_name, and model_path consistently, and create directories.
    """
    # 强制转换为列表 / Force conversion to a list
    if isinstance(options_list, dict):
        options_list = [options_list]
    assert len(options_list) > 0, "options_list should not be empty"

    # 用第一个阶段的 options 生成 save_name 和 log_name / Generate save_name and log_name from the first stage options
    ref_opt = options_list[0]
    model_name = ref_opt['model_name']
    save_name = f"{model_name}_{train_mode}_{current_time}"
    log_name = os.path.join(ref_opt['path_to_env'], ref_opt['log_path'], save_name + ".log")

    for opt in options_list:
        opt['save_name'] = save_name
        opt['log_name'] = log_name

        # 若 model_path 是完整路径则直接使用，否则补全路径 / Use model_path directly if it is absolute; otherwise complete the path
        if not os.path.isabs(opt['model_path']):
            opt['model_path'] = os.path.join(opt['path_to_env'], opt['model_path'], save_name, opt['stage_name'])

        os.makedirs(opt['model_path'], exist_ok=True)

        option_md_path = os.path.join(opt['model_path'], 'options.md')
        save_options_markdown(opt, option_md_path, f"{opt['stage_name']} Settings")

    # 日志只配置一次（防止重复配置）/ Configure logging only once to avoid duplicated handlers
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(message)s')
    print(f"[INFO] Logging setup complete at: {log_name}")


if __name__ == "__main__":
    # --------- 控制训练模式 / Control training mode ---------
    model_name = 'slapnet'  # 可以通过 argparse 或配置文件获取 / Can be obtained through argparse or a configuration file
    stage1_options, stage2_options, stage3_options = load_options_by_model(model_name)
    train_mode = 'stage1'  # 可选值：'stage1', 'stage2', 'both', 'stage2_resume', 'stage3_resume', 'all' / Available options: 'stage1', 'stage2', 'both', 'stage2_resume', 'stage3_resume', 'all'
    resume_ckpt_path = ''  # 若使用 resume，这里填入预训练模型路径 / If resume is used, specify the pretrained checkpoint path here
 
    # -- 固定随机种子 / Fix random seed -- #
    if stage1_options['seed'] is not None:  # 固定随机因子 / Fix random factors
        seed_everything(stage1_options['seed'])
        print(f'Seed {stage1_options["seed"]} fixed.')
    else:  # 随机因子不固定时 / When random factors are not fixed
        torch.backends.cudnn.benchmark = True  # 为 GPU 选择与给定输入尺寸匹配且性能较优的内核 / Select the best-performing CUDA kernels for the given input size

    # -- 文件名设置 / Filename setup -- #
    current_time =  datetime.now().strftime("%Y-%m-%d_%H-%M")
    for cfg in (stage1_options, stage2_options, stage3_options):
        cfg['path_to_npy'] = os.path.join(stage1_options['path_to_env'], stage1_options['npy_path'])  # npy 存储位置 / Storage location for npy files

    # -- GPU 设置 / GPU setup -- #
    if torch.cuda.is_available():
        print('GPU available!')
        print('Total number of available devices: ', torch.cuda.device_count())
        device = torch.device("cuda")
    else:
        print('GPU not available.')
        device = torch.device('cpu')
    for cfg in (stage1_options, stage2_options, stage3_options): cfg['device'] = device
    print('Using device: ', device)

    # -- 训练 / Training
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # 多阶段训练模式使用统一配置（如 both、all）/ Use unified settings for multi-stage training modes such as both and all
    if train_mode in ['both', 'all']:
        all_opts = [stage1_options, stage2_options] + ([stage3_options] if train_mode == 'all' else [])
        setup_training_stage(all_opts, train_mode, current_time)

        best_stage1, epoch_offset1 = run_stage(stage1_options)
        best_stage2, epoch_offset2 = run_stage(stage2_options, resume_ckpt=best_stage1, epoch_offset=epoch_offset1)
        if train_mode == 'all':
            run_stage(stage3_options, resume_ckpt=best_stage2, epoch_offset=epoch_offset1 + epoch_offset2)


    elif train_mode == 'stage1':
        setup_training_stage(stage1_options, 'stage1', current_time)
        run_stage(stage1_options)

    elif train_mode == 'stage2':
        setup_training_stage(stage2_options, 'stage2', current_time)
        run_stage(stage2_options)

    elif train_mode == 'stage2_resume':
        setup_training_stage(stage2_options, 'stage2_resume', current_time)
        run_stage(stage2_options, resume_ckpt=resume_ckpt_path, epoch_offset=60)
        
    elif train_mode == 'stage3':
        setup_training_stage(stage3_options, 'stage3', current_time)
        run_stage(stage3_options)

    elif train_mode == 'stage3_resume':
        setup_training_stage(stage3_options, 'stage3_resume', current_time)
        run_stage(stage3_options, resume_ckpt=resume_ckpt_path, epoch_offset=120)
        
        # 新增：stage1 -> stage3 / Added mode: stage1 -> stage3
    elif train_mode == 'st1_then_st3':
        assert stage3_options is not None, "options 中缺少 stage3_options，无法运行 stage3"
        # 统一创建目录与日志（建议把两个阶段放同一组 save_name 下）/ Create directories and logs together; recommended to place both stages under the same save_name
        setup_training_stage([stage1_options, stage3_options], train_mode, current_time)

        best_stage1, epoch_offset1 = run_stage(stage1_options)
        # stage3 从 stage1 的 best ckpt 初始化 / Initialize stage3 from the best checkpoint of stage1
        run_stage(stage3_options, resume_ckpt=best_stage1, epoch_offset=epoch_offset1)

    else:
        raise ValueError(f"Invalid train_mode: {train_mode}")

    print("Training Complete.")
