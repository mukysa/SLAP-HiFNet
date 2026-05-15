# -*- coding: utf-8 -*-
# -- 内置模块 / Built-in modules -- #
import os
import shutil
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 替换为保留的显卡编号 / Replace with the reserved GPU ID

# -- 第三方模块 / Third-party modules -- #
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
from collections import defaultdict
from numpy.ma import masked_where
from skimage import exposure
import copy
import importlib

# -- 自定义模块 / Project-specific modules -- #
# 工具函数 / Utility functions
from utils.split_train_val import split_randomly, get_manual_dataset_lists, get_manual_list_all # 划分训练验证的相关函数 / Utilities for train/validation/test split
from utils.functions import r2_metric, f1_metric, miou_metric, slice_middle_classes, _f1_group
from utils.utils import _fmt_metric
from dataloaders.loaders import SarAmsrSenceValTestDataset

DATA_CACHE_DIR = os.path.join('/share/home/u10109/data/yushi/CTNet/model_results', 'data_cache')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# 控制测试阶段 / Control testing stages
run_stage1 = True
run_stage2 = True
run_stage3 = False
run_stage2_on_stage1 = True
run_stage3_on_stage1 = False
run_stage1_on_stage2 = True
run_get_manual_label = True


# === 合并统计开关 / Switches for merged metric calculation ===
merge_stage2_and_st2onst1 = True  # True: 只做合并统计 / only calculate merged metrics; False: 保持分阶段各算各的 / calculate each stage separately
merge_stage3_and_st3onst1 = False

# -- 模型路径设置 / Model path settings -- #
model_name = 'slapnet'  # ← 可替换为其他模型名 / Can be replaced with other model names, e.g., simpleicenet, unet, ss_4cat, ss_p4cat, rr_4cat, swin, 2g2m

model_path = ''  # Replace with the training folder you want to test

model_type = 'last_best_model' # Types of models to be tested: best_model, last_best_model, last_model

# inference = True
# cal_score = True
# visual = False

inference = False
cal_score = False
visual = True

clean = False

def build_model(options):
    model_name = options['model_name']
    
    # -- 单模态模型 / Single-modality models -- #
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

    # -- 输入级融合 / Early fusion -- #
    # CNN 模型 / CNN models
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
    # Transformer 模型 / Transformer models
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
    elif model_name == 'rr_4cat':  # 4组实验设置 / Four settings: SsSs, StSt, StSs, SsSt
        from models.rr_4cat import RR4cat
        model = RR4cat(options)
    elif model_name == 'ss_4cat':
        from models.ss_4cat import SS4cat
        model = SS4cat(options)
    
    # -- 本文方法 / Ours -- #    
    elif model_name == 'slapnet':
        from models.slapnet import SLAPNet
        model = SLAPNet(options)
        
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")

    return model


def _clean_test_results(root_path: str, clear_all: bool = True, model_type: str = None):
    """
    Safely clean test_results.
    安全清理 test_results：
      - clear_all=True: remove the entire root_path/test_results directory. / 直接删除 root_path/test_results 整个目录。
      - clear_all=False with model_type: remove only root_path/test_results/<model_type>. / 当 clear_all=False 且给定 model_type 时，仅删除 root_path/test_results/<model_type>。
    This function should only be called when inference=True. / 仅在 inference=True 时调用。
    """
    if not root_path or not isinstance(root_path, str):
        print(f"[CLEAN] Skip: invalid root_path={root_path}")
        return
    tr = os.path.join(root_path, 'test_results')
    try:
        if clear_all:
            # 防御性检查，确保路径名正确，避免误删上级目录 / Defensive check to avoid accidentally deleting parent directories
            if os.path.basename(tr) != 'test_results':
                print(f"[CLEAN] Safety check failed for path: {tr}")
                return
            if os.path.exists(tr):
                shutil.rmtree(tr, ignore_errors=True)
                print(f"[CLEAN] Removed: {tr}")
        else:
            if model_type is None:
                print("[CLEAN] Skip: model_type is None while clear_all=False")
                return
            sub = os.path.join(tr, model_type)
            if os.path.exists(sub):
                shutil.rmtree(sub, ignore_errors=True)
                print(f"[CLEAN] Removed: {sub}")
    except Exception as e:
        print(f"[CLEAN][WARN] Failed to clean {tr}: {e}")


def load_options_by_model(model_name: str):
    """
    Dynamically import the configuration from the options module according to the model name.
    根据模型名从 options 模块动态导入配置。
    The corresponding options script should contain stage1_options, stage2_options, and stage3_options when needed.
    对应的 options 脚本应在需要时包含 stage1_options、stage2_options 和 stage3_options。
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


def test_stage(test_options, model_load_path, model_type, save_root_path):
    print(f"\nRunning {test_options['stage_name']} test (sliding window)...")

    # ===== 1. 获取测试列表 / Get the test file list =====
    if 'custom_test_list' in test_options:
        test_files = test_options['custom_test_list']
    else:
        if test_options['stage_name'] in ['s1_pretrain', 'st2_on_st1', 'st3_on_st1']:
            json_file_path = os.path.join(test_options['path_to_env'], test_options.get('test_json_path'))
            test_files, _ = split_randomly(json_file_path, val_count=0)
        else:
            json_file_path = os.path.join(test_options['path_to_env'], test_options.get('test_json_path'))
            _, _, test_files = get_manual_dataset_lists(json_file_path, fold_id=test_options['fold_id'])

    print('Test set initialisation complete.')

    # ===== 2. 加载数据集 / Load the dataset =====
    dataset = SarAmsrSenceValTestDataset(test_options, test_files)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    print('Dataloader initialisation complete.')

    # ===== 3. 初始化模型并加载权重 / Initialize the model and load weights =====
    net = build_model(test_options)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net = net.to(test_options['device'])

    # 模型加载 / Model loading
    ckpt_path = os.path.join(model_load_path, model_type)
    ckpt = torch.load(ckpt_path, map_location=test_options['device'])
    net.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f'Model loaded from: {ckpt_path}')

    net.eval()
    # 保存目录 / Output directory
    save_dir = os.path.join(save_root_path, 'test_results', model_type)
    os.makedirs(save_dir, exist_ok=True)

    # ===== 4. 推理并保存结果 / Run inference and save results =====
    with torch.no_grad():
        tq = tqdm(dataloader, desc=f"Testing {test_options['stage_name']}", ncols=100)
        for batch in tq:
            hr_windows, label_dict, mask, window_positions, scene_shape, pad_shape, sar, name = batch
            H, W = scene_shape
            ws = test_options['test_window_size']
            device = test_options['device']

            count_matrix = torch.zeros(pad_shape, dtype=torch.int32)
            logits_sum_dict = {}
            for label_key in test_options['labels']:
                ncls = 2 if label_key == 'label_manual' else test_options['nclass']
                logits_sum_dict[label_key] = torch.zeros((1, ncls, *pad_shape), dtype=torch.float32).to(device)

            for x, (i, j) in zip(hr_windows, window_positions):
                x = x.unsqueeze(0).to(device)
                with torch.cuda.amp.autocast():
                    outputs_dict = net(x, mode=test_options['net_mode'], use_cnn_iw_head=True, use_trans_iw_head=True)
                # === [MODIFIED-1] 处理 outputs_dict 中的每个输出，不受 test_options['labels'] 限制 / Process each output in outputs_dict regardless of test_options['labels'] ===
                for label_key in outputs_dict.keys():
                    if label_key not in ['label_80_sic', 'label_manual']:
                        continue
                    # === [MODIFIED-3] 动态初始化 logits_sum_dict 中的键 / Dynamically initialize keys in logits_sum_dict ===
                    if label_key not in logits_sum_dict:
                        ncls = 2 if label_key == 'label_manual' else test_options['nclass']
                        logits_sum_dict[label_key] = torch.zeros((1, ncls, *pad_shape), dtype=torch.float32).to(device)

                    logits_sum_dict[label_key][:, :, i:i+ws, j:j+ws] += outputs_dict[label_key][0].detach()

                count_matrix[i:i+ws, j:j+ws] += 1

            count_matrix = torch.clamp(count_matrix, min=1).to(device)
            file_base = os.path.basename(name)

            # 保存预测结果 / Save prediction results
            for label_key in logits_sum_dict.keys():
                if label_key not in logits_sum_dict:
                    continue
                logits_avg = logits_sum_dict[label_key] / count_matrix.unsqueeze(0).unsqueeze(0)
                logits_avg = logits_avg[:, :, :H, :W]
                output = torch.argmax(logits_avg, dim=1).squeeze(0).cpu().numpy()
                # 屏蔽无效区域 / Mask invalid regions
                mask = mask[:H, :W]
                output = output[:H, :W]
                output[mask] = 255  # 屏蔽无效区域 / Mask invalid regions
                np.save(os.path.join(save_dir, f'{file_base}_{label_key}_output.npy'), output)
            #     # === [MODIFIED-2] 仅当 label_dict 中存在对应标签时保存 chart / Save chart only when the label exists in label_dict ===
            #     if label_key in label_dict:
            #         np.save(os.path.join(save_dir, f'{file_base}_{label_key}_chart.npy'), label_dict[label_key])
            # np.save(os.path.join(save_dir, f'{file_base}_mask.npy'), mask)

            # # 保存 SAR 图像 / Save SAR image
            # np.save(os.path.join(save_dir, f'{file_base}_sar.npy'), sar)
            # ===== 通用数据写入 data_cache / Write shared data to data_cache =====
            cache_base = os.path.join(DATA_CACHE_DIR, file_base)

            # SAR 数据 / SAR data
            sar_path = f"{cache_base}_sar.npy"
            if not os.path.exists(sar_path):
                np.save(sar_path, sar)

            # 掩膜 / Mask
            mask_path = f"{cache_base}_mask.npy"
            if not os.path.exists(mask_path):
                np.save(mask_path, mask[:H, :W])

            # 标签图，仅在 GT 存在时保存 / Label charts, saved only when GT is available
            for label_key in ['label_80_sic', 'label_manual']:
                if label_key in label_dict:
                    chart_path = f"{cache_base}_{label_key}_chart.npy"
                    if not os.path.exists(chart_path):
                        np.save(chart_path, label_dict[label_key])


    print("Inference complete and results saved!")
    

def calculate_score_from_npy(save_dir, test_options):
    label_keys = test_options['labels']
    print("Calculating scores from:", save_dir)

    # 初始化结果容器 / Initialize result containers
    pred_dict = defaultdict(list)
    chart_dict = defaultdict(list)

    # === 遍历每个场景 base / Iterate over each scene base ===
    # base_list = []
    # for file in os.listdir(DATA_CACHE_DIR):
    #     if file.endswith('_sar.npy'):
    #         base = file[:-len('_sar.npy')]
    #         base_list.append(base)
    # 只从 test_results 下现有的预测文件确定 base / Determine base names only from prediction files under test_results
    base_list = []
    for f in os.listdir(save_dir):
        if f.endswith('_label_80_sic_output.npy') or f.endswith('_label_manual_output.npy'):
            base = f.split('_label_')[0]
            if base not in base_list:
                base_list.append(base)


    for base in base_list:
        mask_path = os.path.join(DATA_CACHE_DIR, f'{base}_mask.npy')
        if not os.path.exists(mask_path):
            print(f"[SKIP] Missing mask for {base}")
            continue
        mask = np.load(mask_path)
        valid = ~mask  # 注意是布尔型掩膜 / Boolean mask is expected here

        for label_key in label_keys:
            chart_path = os.path.join(DATA_CACHE_DIR, f'{base}_{label_key}_chart.npy')
            output_path = os.path.join(save_dir, f'{base}_{label_key}_output.npy')
            if not (os.path.exists(chart_path) and os.path.exists(output_path)):
                print(f"[SKIP] Missing files for {base} - {label_key}")
                continue

            chart = np.load(chart_path)
            output = np.load(output_path)

            if chart.shape != output.shape or chart.shape != mask.shape:
                print(f"[SKIP] Shape mismatch for {base} - {label_key}")
                continue

            # 有效区域 / Valid regions
            pred_dict[label_key].append(output[valid])
            chart_dict[label_key].append(chart[valid])

    # === 计算指标 / Calculate metrics ===
    # 可配置参数：SIC 中间类别范围与 BF1 容差 / Configurable parameters: SIC middle-class range and BF1 tolerance
    sic_mid_lo = 1
    sic_mid_hi = 8
    
    results = {}
    for label_key in label_keys:
        if label_key not in pred_dict or not pred_dict[label_key]:
            print(f"No valid predictions for {label_key}")
            continue

        y_pred = np.concatenate(pred_dict[label_key])
        y_true = np.concatenate(chart_dict[label_key])

        if label_key == 'label_manual':
            f1 = f1_metric(y_true, y_pred)
            miou, _ = miou_metric(y_true, y_pred, n_classes=2)
            # 计算边界容差 F1（对二值图）/ Calculate boundary-tolerant F1 for binary maps
            results[label_key] = {'F1': f1, 'mIoU': miou}
        else:  # 'label_80_sic'
            r2 = r2_metric(y_true, y_pred)
            f1 = f1_metric(y_true, y_pred)

            # 中间类别（1-8）/ Middle classes (1-8)
            y_true_mid, y_pred_mid = slice_middle_classes(
                y_true, y_pred, nclass=test_options['nclass'], lo=sic_mid_lo, hi=sic_mid_hi
            )
            if y_true_mid.size > 0:
                r2_mid = r2_metric(y_true_mid, y_pred_mid)
                f1_mid = f1_metric(y_true_mid, y_pred_mid)
            else:
                r2_mid, f1_mid = np.nan, np.nan

            # 0 类（水）与高密度冰（9-10）的二分类 F1 / Binary F1 for water class 0 and high-concentration ice classes 9-10
            f1_water_0   = _f1_group(y_true, y_pred, [0])
            f1_high_9_10 = _f1_group(y_true, y_pred, [9, 10])

            results[label_key] = {
                'R2': r2, f'R2_mid[{sic_mid_lo}-{sic_mid_hi}]': r2_mid,
                'F1': f1, 'F1_water_0': f1_water_0,
                f'F1_mid[{sic_mid_lo}-{sic_mid_hi}]': f1_mid,
                'F1_high_9_10': f1_high_9_10
            }

    # === 打印并保存结果 / Print and save results ===
    out_txt = os.path.join(save_dir, 'metrics_result.txt')
    with open(out_txt, 'w') as f:
        for k, v in results.items():
            print(f"\n=== Metrics for {k} ===")
            f.write(f"=== Metrics for {k} ===\n")
            for metric_name, value in v.items():
                fv = _fmt_metric(metric_name, value)
                print(f"{metric_name}: {fv}")
                f.write(f"{metric_name}: {fv}\n")
    print(f"\nSaved metrics to {out_txt}")
    
    
def calculate_merged_stage2_metrics(st2_dir, st2_opts, st2onst1_dir, st2onst1_opts, out_dir):
    """
    Calculate merged metrics.
    合并统计：
    - SIC: concatenate predictions and ground truth from stage2_dir and st2onst1_dir for unified metrics (R2, R2_mid, overall F1, and grouped F1).
      SIC：将 stage2_dir 与 st2onst1_dir 的预测和真值拼接后统一统计（R2、R2_mid、总体 F1 及分组 F1）。
    - IW: calculate metrics only from stage2_dir (F1 and mIoU).
      IW：仅从 stage2_dir 统计（F1 和 mIoU）。
    Save results to out_dir/metrics_result.txt.
    结果保存到 out_dir/metrics_result.txt。
    """
    os.makedirs(out_dir, exist_ok=True)

    def collect_for_dir(save_dir, label_key, nclass):
        pred_list, chart_list = [], []

        # 只统计当前 save_dir 中实际存在该 label 预测的场景 / Use only scenes with existing predictions for this label in the current save_dir
        base_list = []
        suffix = f'_{label_key}_output.npy'
        for f in os.listdir(save_dir):
            if f.endswith(suffix):
                base = f[:-len(suffix)]
                if base not in base_list:
                    base_list.append(base)

        for base in base_list:
            mask_p  = os.path.join(DATA_CACHE_DIR, f'{base}_mask.npy')
            chart_p = os.path.join(DATA_CACHE_DIR, f'{base}_{label_key}_chart.npy')
            pred_p  = os.path.join(save_dir, f'{base}_{label_key}_output.npy')
            if not (os.path.exists(mask_p) and os.path.exists(chart_p) and os.path.exists(pred_p)):
                continue

            mask  = np.load(mask_p)        # bool
            chart = np.load(chart_p)
            pred  = np.load(pred_p)

            if chart.shape != pred.shape or chart.shape != mask.shape:
                continue

            valid = ~mask
            pred_list.append(pred[valid])
            chart_list.append(chart[valid])

        return pred_list, chart_list

    # ====== 1) SIC 合并（stage2 + st2_on_st1）/ Merge SIC results (stage2 + st2_on_st1) ======
    sic_key = 'label_80_sic'
    nclass_sic = st2_opts['nclass']  # 两边 nclass 应一致 / nclass should be consistent between the two stages
    pred_sic_all, chart_sic_all = [], []

    # stage2 的 SIC / SIC from stage2
    p_s2, c_s2 = collect_for_dir(st2_dir, sic_key, nclass_sic)
    pred_sic_all += p_s2
    chart_sic_all += c_s2
    # st2_on_st1 的 SIC / SIC from st2_on_st1
    p_s2on1, c_s2on1 = collect_for_dir(st2onst1_dir, sic_key, nclass_sic)
    pred_sic_all += p_s2on1
    chart_sic_all += c_s2on1

    results = {}

    if len(pred_sic_all) > 0:
        y_pred = np.concatenate(pred_sic_all)
        y_true = np.concatenate(chart_sic_all)

        # 全类别 R2 / F1 / Overall R2 and F1
        r2_all = r2_metric(y_true, y_pred)
        f1_all = f1_metric(y_true, y_pred)

        # 中间类别段（默认 1-8）/ Middle-class range (default: 1-8)
        sic_mid_lo, sic_mid_hi = 1, 8
        y_true_mid, y_pred_mid = slice_middle_classes(
            y_true, y_pred, nclass=nclass_sic, lo=sic_mid_lo, hi=sic_mid_hi
        )
        if y_true_mid.size > 0:
            r2_mid = r2_metric(y_true_mid, y_pred_mid)
            f1_mid = f1_metric(y_true_mid, y_pred_mid)
        else:
            r2_mid, f1_mid = np.nan, np.nan

        # 水（0）/ 高密度冰（9-10）的二分类 F1 / Binary F1 for water (0) and high-concentration ice (9-10)
        f1_water_0   = _f1_group(y_true, y_pred, [0])
        f1_high_9_10 = _f1_group(y_true, y_pred, [9, 10])

        results['SIC(merged s2 + st2_on_st1)'] = {
            'R2': r2_all,
            f'R2_mid[{sic_mid_lo}-{sic_mid_hi}]': r2_mid,
            'F1': f1_all,
            'F1_water_0': f1_water_0,
            f'F1_mid[{sic_mid_lo}-{sic_mid_hi}]': f1_mid,
            'F1_high_9_10': f1_high_9_10
        }
    else:
        print('[WARN] No valid SIC pixels are available for merged metric calculation.')

    # ====== 2) IW 指标（仅 stage2_dir）/ IW metrics (stage2_dir only) ======
    iw_key = 'label_manual'
    pred_iw, chart_iw = collect_for_dir(st2_dir, iw_key, nclass=2)
    if len(pred_iw) > 0:
        y_pred_iw = np.concatenate(pred_iw)
        y_true_iw = np.concatenate(chart_iw)
        iw_f1  = f1_metric(y_true_iw, y_pred_iw)
        iw_mIoU, _ = miou_metric(y_true_iw, y_pred_iw, n_classes=2)
        results['IW(stage2 only)'] = {'F1': iw_f1, 'mIoU': iw_mIoU}
    else:
        print('[WARN] No valid IW pixels were found in the stage2 directory for metric calculation.')

    # ====== 3) 打印并保存 / Print and save ======
    out_txt = os.path.join(out_dir, 'metrics_result.txt')
    with open(out_txt, 'w') as f:
        for k, v in results.items():
            print(f"\n=== Metrics for {k} ===")
            f.write(f"=== Metrics for {k} ===\n")
            for metric_name, value in v.items():
                fv = _fmt_metric(metric_name, value)
                print(f"{metric_name}: {fv}")
                f.write(f"{metric_name}: {fv}\n")
    print(f"\n[MERGED] Saved metrics to {out_txt}")


def visualize_from_npy(test_options, save_dir, model_path, model_type):
    """
    Read the npy files of each scene, generate visualizations, and calculate per-scene metrics.
    读取每个场景的 npy 文件，生成可视化图像，并计算该场景指标。
      - SIC: R2 and F1. / SIC：R2 和 F1。
      - IW: F1 and mIoU. / IW：F1 和 mIoU。
    Metrics are written to the figure title, summarized in visualization/<model_type>/per_scene_metrics.csv, and printed to the terminal.
    指标会写入图像标题，汇总到 visualization/<model_type>/per_scene_metrics.csv，并打印到终端。
    """
    vis_dir = os.path.join(model_path, 'visualization', model_type)
    os.makedirs(vis_dir, exist_ok=True)

    # base_list = [f[:-len('_sar.npy')] for f in os.listdir(DATA_CACHE_DIR) if f.endswith('_sar.npy')]
    # 只从 test_results 下现有的预测文件确定 base / Determine base names only from prediction files under test_results
    base_list = []
    for f in os.listdir(save_dir):
        if f.endswith('_label_80_sic_output.npy') or f.endswith('_label_manual_output.npy'):
            base = f.split('_label_')[0]
            if base not in base_list:
                base_list.append(base)

    # 准备 CSV 输出（覆盖写）/ Prepare CSV output and overwrite existing files
    csv_path = os.path.join(vis_dir, "per_scene_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write("scene,SIC_R2,SIC_F1,IW_F1,IW_mIoU\n")

    def _fmt(v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "nan"
            return f"{float(v):.3f}"
        except Exception:
            return "nan"

    for base in base_list:
        # 先默认设为 NaN，以便在文件缺失时安全跳过 / Initialize as NaN to handle missing files gracefully
        sic_r2 = np.nan
        sic_f1 = np.nan
        iw_f1  = np.nan
        iw_miou = np.nan

        sar_path  = os.path.join(DATA_CACHE_DIR, f"{base}_sar.npy")
        if not os.path.exists(sar_path):
            print(f"[SKIP] Missing SAR for {base}")
            continue
        sar = np.load(sar_path)
        sar_hh_eq = exposure.equalize_hist(sar[0])
        sar_hv_eq = exposure.equalize_hist(sar[1])

        # 加载掩膜 / Load mask
        mask_path = os.path.join(DATA_CACHE_DIR, f"{base}_mask.npy")
        if not os.path.exists(mask_path):
            print(f"[SKIP] Missing mask for {base}")
            continue
        mask = np.load(mask_path)  # bool 类型 / Boolean type
        H, W = mask.shape
        blank_img = np.ones((H, W)) * 255
        blank_masked = masked_where(blank_img == 255, blank_img)

        # 先尝试加载两类任务的 GT 与 Pred 并计算指标 / Try loading GT and predictions for both tasks before metric calculation
        # --- SIC ---
        sic_chart_p = os.path.join(DATA_CACHE_DIR, f"{base}_label_80_sic_chart.npy")
        sic_pred_p  = os.path.join(save_dir, f"{base}_label_80_sic_output.npy")
        if os.path.exists(sic_chart_p) and os.path.exists(sic_pred_p):
            sic_chart = np.load(sic_chart_p)
            sic_pred  = np.load(sic_pred_p)
            if sic_chart.shape == sic_pred.shape == mask.shape:
                valid = ~mask
                y_true_sic = sic_chart[valid]
                y_pred_sic = sic_pred[valid]
                # 指标 / Metrics
                try:
                    sic_r2 = r2_metric(y_true_sic, y_pred_sic)
                except Exception:
                    sic_r2 = np.nan
                try:
                    sic_f1 = f1_metric(y_true_sic, y_pred_sic)
                except Exception:
                    sic_f1 = np.nan

        # --- IW ---
        iw_chart_p  = os.path.join(DATA_CACHE_DIR, f"{base}_label_manual_chart.npy")
        iw_pred_p  = os.path.join(save_dir, f"{base}_label_manual_output.npy")
        if os.path.exists(iw_chart_p) and os.path.exists(iw_pred_p):
            iw_chart = np.load(iw_chart_p)
            iw_pred  = np.load(iw_pred_p)
            if iw_chart.shape == iw_pred.shape == mask.shape:
                valid = ~mask
                y_true_iw = iw_chart[valid]
                y_pred_iw = iw_pred[valid]
                # 指标 / Metrics
                try:
                    iw_f1 = f1_metric(y_true_iw, y_pred_iw)
                except Exception:
                    iw_f1 = np.nan
                try:
                    iw_miou, _ = miou_metric(y_true_iw, y_pred_iw, n_classes=2)
                except Exception:
                    iw_miou = np.nan

        # === 画图 / Plot visualizations ===
        fig, axes = plt.subplots(2, 4, figsize=(24, 10), dpi=600)
        fig.suptitle(
            f"{base} — model: {model_type} | "
            f"SIC[R2={_fmt(sic_r2)}, F1={_fmt(sic_f1)}] · "
            f"IW[F1={_fmt(iw_f1)}, mIoU={_fmt(iw_miou)}]",
            fontsize=14
        )

        for ax in axes.flat:
            ax.axis('off')
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   fill=False, edgecolor='black', linewidth=2))

        # SAR 两个波段单独显示 / Display the two SAR bands separately
        axes[0, 0].imshow(sar_hh_eq, cmap='gray')
        axes[0, 0].set_title("SAR HH", fontsize=12)

        axes[1, 0].imshow(sar_hv_eq, cmap='gray')
        axes[1, 0].set_title("SAR HV", fontsize=12)

        # 逐任务可视化 / Task-wise visualization
        for label_key, row in zip(['label_80_sic', 'label_manual'], [0, 1]):
            chart_path = os.path.join(DATA_CACHE_DIR, f"{base}_{label_key}_chart.npy")
            pred_path = os.path.join(save_dir, f"{base}_{label_key}_output.npy")

            if not (os.path.exists(chart_path) and os.path.exists(pred_path)):
                # 缺失时填充空白图 / Fill missing panels with blank images
                for col, name in zip([1, 2, 3], ['Pred', 'GT', 'Diff']):
                    axes[row, col].imshow(blank_masked, cmap='gray')
                    axes[row, col].set_title(f"{name} {'SIC' if label_key=='label_80_sic' else 'IW'} (missing)", fontsize=12)
                continue

            chart = np.load(chart_path).copy()
            pred = np.load(pred_path).copy()

            if chart.shape != pred.shape or chart.shape != mask.shape:
                print(f"[SKIP] Shape mismatch for {base} - {label_key}")
                for col, name in zip([1, 2, 3], ['Pred', 'GT', 'Diff']):
                    axes[row, col].imshow(blank_masked, cmap='gray')
                    axes[row, col].set_title(f"{name} {'SIC' if label_key=='label_80_sic' else 'IW'} (shape mismatch)", fontsize=12)
                continue

            pred[mask] = 255
            chart[mask] = 255
            diff = np.abs(pred - chart)
            diff[mask] = 255

            pred_masked = masked_where(pred == 255, pred)
            chart_masked = masked_where(chart == 255, chart)
            diff_masked = masked_where(diff == 255, diff)

            if label_key == 'label_80_sic':
                cmap = plt.cm.viridis
                cmap.set_bad(alpha=0.0)

                axes[row, 1].imshow(pred_masked, cmap=cmap, vmin=0, vmax=test_options['nclass'] - 1)
                axes[row, 1].set_title("Pred SIC", fontsize=12)

                axes[row, 2].imshow(chart_masked, cmap=cmap, vmin=0, vmax=test_options['nclass'] - 1)
                axes[row, 2].set_title("GT SIC", fontsize=12)

                diff_cmap = plt.cm.plasma
                diff_cmap.set_bad(alpha=0.0)
                axes[row, 3].imshow(diff_masked, cmap=diff_cmap, vmin=0, vmax=test_options['nclass'] - 1)
                axes[row, 3].set_title("Diff SIC", fontsize=12)

            elif label_key == 'label_manual':
                cmap = colors.ListedColormap(['blue', '#FAEBD7'])  # 蓝=水，米黄=冰 / Blue=water, beige=ice
                norm = colors.BoundaryNorm([0, 1, 2], cmap.N)

                axes[row, 1].imshow(pred_masked, cmap=cmap, norm=norm)
                axes[row, 1].set_title("Pred IW", fontsize=12)

                axes[row, 2].imshow(chart_masked, cmap=cmap, norm=norm)
                axes[row, 2].set_title("GT IW", fontsize=12)

                axes[row, 3].imshow(diff_masked, cmap='plasma', vmin=0, vmax=1)
                axes[row, 3].set_title("Diff IW", fontsize=12)
                
        # === [恢复原逻辑] 特殊显示 IW 预测图，即使 labels 中没有 / Restore the original logic: show IW prediction even if it is absent from labels ===
        if test_options.get("stage_name") == "st2_on_st1":
            iw_pred_path = os.path.join(save_dir, f"{base}_label_manual_output.npy")
            if os.path.exists(iw_pred_path):
                iw_pred = np.load(iw_pred_path).copy()
                iw_pred[mask] = 255
                iw_pred_masked = masked_where(iw_pred == 255, iw_pred)

                cmap = colors.ListedColormap(['blue', '#FAEBD7'])  # 蓝=水，米黄=冰 / Blue=water, beige=ice
                norm = colors.BoundaryNorm([0, 1, 2], cmap.N)

                # 用下方一行第二列显示 / Display it in the second column of the lower row
                axes[1, 1].imshow(iw_pred_masked, cmap=cmap, norm=norm)
                axes[1, 1].set_title("Pred IW (forced)", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = os.path.join(vis_dir, f"{base}_vis.png")
        plt.savefig(out_png)
        plt.close()

        # 终端打印并追加到 CSV / Print to terminal and append to CSV
        print(f"[{base}]  SIC: R2={_fmt(sic_r2)}, F1={_fmt(sic_f1)} | IW: F1={_fmt(iw_f1)}, mIoU={_fmt(iw_miou)}")
        with open(csv_path, "a", encoding="utf-8") as fcsv:
            fcsv.write(f"{base},{_fmt(sic_r2)},{_fmt(sic_f1)},{_fmt(iw_f1)},{_fmt(iw_miou)}\n")

    print(f"[VIS] Per-scene metrics saved to: {csv_path}")


if __name__ == "__main__": 
    # 获取设置 / Load settings
    stage1_options, stage2_options, stage3_options = load_options_by_model(model_name)
    
    # 拷贝生成新的 options，避免直接修改原始配置 / Deep-copy options to avoid modifying the original configurations
    stage2_on_stage1_options = copy.deepcopy(stage2_options)
    stage1_on_stage2_options = copy.deepcopy(stage1_options)
    stage3_on_stage1_options = copy.deepcopy(stage3_options)
    # 设置新的 stage_name / Set new stage_name values
    stage2_on_stage1_options['stage_name'] = 'st2_on_st1'
    stage1_on_stage2_options['stage_name'] = 'st1_on_st2'
    stage3_on_stage1_options['stage_name'] = 'st3_on_st1'
    # 修改 labels 和 json 路径 / Modify labels and JSON paths
    stage2_on_stage1_options['labels'] = ['label_80_sic']
    stage1_on_stage2_options['labels'] = ['label_80_sic']
    stage3_on_stage1_options['labels'] = ['label_80_sic']
    # 设置新的数据列表 / Set new data lists
    stage2_on_stage1_options['path_to_test_val_data'] = stage1_options['path_to_test_val_data']
    stage2_on_stage1_options['test_json_path'] = stage1_options['test_json_path']  # 使用 stage1 的数据列表 / Use the stage1 data list
    stage1_on_stage2_options['path_to_test_val_data'] = stage2_options['path_to_test_val_data']
    stage1_on_stage2_options['test_json_path'] = stage2_options['json_path']  # 使用 stage2 的数据列表 / Use the stage2 data list
    stage3_on_stage1_options['path_to_test_val_data'] = stage1_options['path_to_test_val_data']
    stage3_on_stage1_options['test_json_path'] = stage1_options['test_json_path']  # 使用 stage1 的数据列表 / Use the stage1 data list
    # 其他更新 / Other updates
    stage1_on_stage2_options['fold_id'] = stage2_options['fold_id']  # 使用 stage2 的 fold 设置 / Use the fold setting from stage2
    
    # 更新模型路径 / Update model paths
    stage1_path = os.path.join(model_path, stage1_options['stage_name'])
    stage2_path = os.path.join(model_path, stage2_options['stage_name'])
    stage3_path = os.path.join(model_path, stage3_options['stage_name'])
    stage2_on_stage1_path = os.path.join(model_path, stage2_on_stage1_options['stage_name'])
    stage1_on_stage2_path = os.path.join(model_path, stage1_on_stage2_options['stage_name'])
    stage3_on_stage1_path = os.path.join(model_path, stage3_on_stage1_options['stage_name'])
    # 确保路径存在 / Ensure output paths exist
    os.makedirs(stage2_on_stage1_path, exist_ok=True)
    os.makedirs(stage1_on_stage2_path, exist_ok=True)
    os.makedirs(stage3_on_stage1_path, exist_ok=True)
    
    # ========== 全人工数据集（用于标签优化）/ Fully manual dataset for label refinement ==========
    if run_get_manual_label:
        stage_manual_all_options = copy.deepcopy(stage2_options)
        stage_manual_all_options['stage_name'] = 'manual_all'
        manual_json_path = stage_manual_all_options['json_path']
        manual_list_all = get_manual_list_all(manual_json_path)
        # 拼接完整路径（不含年份文件夹）/ Build full paths without year subfolders
        stage_manual_all_options['custom_test_list'] = manual_list_all
        # 手动数据集完整推理保存路径 / Output path for full inference on the manual dataset
        manual_all_path = os.path.join(model_path, stage_manual_all_options['stage_name'])
        os.makedirs(manual_all_path, exist_ok=True)
    
    # 推理和保存 / Inference and saving
    if inference:
        if clean:
            # 根据运行开关收集需要清理的写入目录 / Collect output directories to be cleaned according to run switches
            paths_to_clean = []
            if run_stage1:
                paths_to_clean.append(stage1_path)
            if run_stage2:
                paths_to_clean.append(stage2_path)
            if run_stage2_on_stage1:
                paths_to_clean.append(stage2_on_stage1_path)
            if run_stage1_on_stage2:
                paths_to_clean.append(stage1_on_stage2_path)
            if run_stage3:
                paths_to_clean.append(stage3_path)
            if run_get_manual_label:
                # 只有在上面创建了 manual_all_path 的情况下才会有该路径 / This path exists only if manual_all_path was created above
                try:
                    paths_to_clean.append(manual_all_path)
                except NameError:
                    pass

            print("[CLEAN] Inference is ON. Cleaning test_results for write targets ...")
            for p in paths_to_clean:
                # True 表示删除整个 test_results；如只删除当前 model_type 子目录，设置 clear_all=False, model_type=model_type / True removes the entire test_results directory; use clear_all=False with model_type to remove only the current model_type subdirectory
                _clean_test_results(p, clear_all=False, model_type=model_type)
        
        # -- GPU 设置 / GPU settings -- #
        if torch.cuda.is_available():
            print('GPU available!')
            print('Total number of available devices: ', torch.cuda.device_count())
            device = torch.device("cuda")
        else:
            print('GPU not available.')
            device = torch.device('cpu')
        for cfg in (stage1_options, stage2_options, stage2_on_stage1_options, stage1_on_stage2_options, stage3_options, stage3_on_stage1_options): 
            cfg['device'] = device
        print('Using device: ', device)
        
        # 推理时调用 / Run selected inference stages
        if run_stage1:
            test_stage(stage1_options, stage1_path, model_type, stage1_path)
        if run_stage2:
            test_stage(stage2_options, stage2_path, model_type, stage2_path)
        if run_stage2_on_stage1:
            test_stage(stage2_on_stage1_options, stage2_path, model_type, stage2_on_stage1_path)  # 加载 st2 模型并保存到 st2_on_st1 目录 / Load the st2 model and save results to st2_on_st1
        if run_stage1_on_stage2:
            test_stage(stage1_on_stage2_options, stage1_path, model_type, stage1_on_stage2_path)  # 加载 st1 模型并保存到 st1_on_st2 目录 / Load the st1 model and save results to st1_on_st2
        if run_stage3:
            test_stage(stage3_options, stage3_path, model_type, stage3_path)
        if run_stage3_on_stage1:
            test_stage(stage3_on_stage1_options, stage3_path, model_type, stage3_on_stage1_path)

        if run_get_manual_label:
            stage_manual_all_options['device'] = device
            test_stage(stage_manual_all_options, stage2_path, model_type, manual_all_path)

    
    # -- 读取 npy 文件并计算分数 / Read npy files and calculate metrics --
    if cal_score:
        print('Start calculate scores ...')
        if merge_stage2_and_st2onst1 or merge_stage3_and_st3onst1:
            if merge_stage2_and_st2onst1:
                # 仅做合并统计：SIC(stage2 + st2_on_st1) + IW(stage2) / Merged metrics only: SIC(stage2 + st2_on_st1) + IW(stage2)
                save_dir_s2 = os.path.join(stage2_path, 'test_results', model_type)
                save_dir_s2on1 = os.path.join(stage2_on_stage1_path, 'test_results', model_type)
                if os.path.isdir(save_dir_s2) and os.path.isdir(save_dir_s2on1):
                    save_dir_s2 = os.path.join(stage2_path, 'test_results', model_type)
                    save_dir_s2on1 = os.path.join(stage2_on_stage1_path, 'test_results', model_type)
                    out_dir_merged = os.path.join(model_path, 'merged_s2_and_st2_on_st1', model_type)
                    calculate_merged_stage2_metrics(save_dir_s2, stage2_options, save_dir_s2on1, stage2_on_stage1_options, out_dir_merged)
                else:
                    print('[SKIP] stage2 or st2_on_st1 results not found, skip merge_stage2_and_st2onst1')
            if merge_stage3_and_st3onst1:
                # SIC(stage3 + st3_on_st1) + IW(stage3) / Merged metrics for SIC(stage3 + st3_on_st1) and IW(stage3)
                save_dir_s3 = os.path.join(stage3_path, 'test_results', model_type)
                save_dir_s3on1 = os.path.join(stage3_on_stage1_path, 'test_results', model_type)
                if os.path.isdir(save_dir_s3) and os.path.isdir(save_dir_s3on1):
                    save_dir_s3 = os.path.join(stage3_path, 'test_results', model_type)
                    save_dir_s3on1 = os.path.join(stage3_on_stage1_path, 'test_results', model_type)
                    out_dir_merged = os.path.join(model_path, 'merged_s3_and_st3_on_st1', model_type)
                    calculate_merged_stage2_metrics(save_dir_s3, stage3_options, save_dir_s3on1, stage3_on_stage1_options, out_dir_merged)
                else:
                    print('[SKIP] stage3 or st3_on_st1 results not found, skip merge_stage3_and_st3onst1')
        else:
            # 保持原本分阶段统计 / Keep the original per-stage metric calculation
            if run_stage1:
                save_dir = os.path.join(stage1_path, 'test_results', model_type)
                calculate_score_from_npy(save_dir, stage1_options)
            if run_stage2:
                save_dir = os.path.join(stage2_path, 'test_results', model_type)
                calculate_score_from_npy(save_dir, stage2_options)
            if run_stage2_on_stage1:
                save_dir = os.path.join(stage2_on_stage1_path, 'test_results', model_type)
                calculate_score_from_npy(save_dir, stage2_on_stage1_options)
            if run_stage1_on_stage2:
                save_dir = os.path.join(stage1_on_stage2_path, 'test_results', model_type)
                calculate_score_from_npy(save_dir, stage1_on_stage2_options)
            if run_stage3:
                save_dir = os.path.join(stage3_path, 'test_results', model_type)
                calculate_score_from_npy(save_dir, stage3_options)

    # -- 可视化 / Visualization --
    if visual:
        print('Visualizing ...')
        if run_stage1:
            save_dir = os.path.join(stage1_path, 'test_results', model_type)
            visualize_from_npy(stage1_options, save_dir, stage1_path, model_type)
        if run_stage2:
            save_dir = os.path.join(stage2_path, 'test_results', model_type)
            visualize_from_npy(stage2_options, save_dir, stage2_path, model_type)
        if run_stage2_on_stage1:
            save_dir = os.path.join(stage2_on_stage1_path, 'test_results', model_type)
            visualize_from_npy(stage2_on_stage1_options, save_dir, stage2_on_stage1_path, model_type)
        if run_stage1_on_stage2:
            save_dir = os.path.join(stage1_on_stage2_path, 'test_results', model_type)
            visualize_from_npy(stage1_on_stage2_options, save_dir, stage1_on_stage2_path, model_type)
        if run_stage3:
            save_dir = os.path.join(stage3_path, 'test_results', model_type)
            visualize_from_npy(stage3_options, save_dir, stage3_path, model_type)

    print('All processes complete!')
