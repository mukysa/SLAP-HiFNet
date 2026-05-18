# SLAP-HiFNet

本仓库为论文 **SLAP-HiFNet: A Stage-Linked Active-Passive Microwave Hierarchical Fusion Network with Label-Efficient Transfer Learning for Sea Ice Mapping** 的核心代码实现。

SLAP-HiFNet 面向北极海冰制图任务，使用 Sentinel-1 SAR 与 AMSR2 被动微波观测数据进行主动-被动微波融合。该方法结合了分层主动-被动特征融合网络和阶段关联训练策略，先利用易获取的海冰密集度（sea ice concentration, SIC）产品进行预训练，再将学习到的表征迁移到标签稀缺的冰水二分类（ice-water classification, IW）任务中。

（注意，当前项目仍在优化中，因此非核心代码可能仍为未订正中文注释，且尚未支持argparse方式，另外本readme文件也会不断细化）

## 1. 项目简介

本仓库提供 SLAP-HiFNet 及论文中主要对照模型的训练、测试、评估和可视化代码。

主要内容包括：

- 基于 Sentinel-1 SAR 和 AMSR2 的主动-被动微波融合；
- 用于 SIC 预训练和 SIC-IW 联合训练的阶段关联训练流程；
- 支持多尺度特征交互的分层融合结构；
- 支持 SIC 与 IW 双任务预测；
- 包含单源输入、输入级融合、特征级融合等对照网络；
- 提供训练、测试、评价指标计算和可视化脚本。

## 2. 仓库结构

```text
SLAP-HiFNet/
├── dataloaders/          # 数据读取与预处理相关代码
├── dataset/          	  # 数据集json及其生成代码
├── models/               # SLAP-HiFNet 和对照模型定义
├── options/              # 不同模型与实验阶段的配置文件
├── pretrained_weights/   # 预训练权重
├── readme_image/         # 本readme文件的图片
├── utils/                # 损失函数、评价指标、学习率调度器和辅助函数
├── train.py              # 训练入口脚本
├── test.py               # 测试、评价和可视化入口脚本
├── LICENSE
└── README.md
```

## 3. 方法概述

SLAP-HiFNet 采用双分支编码结构，分别提取 Sentinel-1 SAR 和 AMSR2 的多尺度特征。网络在浅层使用轻量门控融合模块，在深层使用注意力融合模块，从而实现不同尺度上的主动-被动微波特征交互。此外，网络在 SIC 分支中引入 AMSR2 先验残差注入模块，以增强 SIC 预测对被动微波观测的利用，同时尽量减少对 IW 结构信息学习的干扰。

训练流程分为两个阶段：

1. **Stage 1：SIC 预训练**

   使用大规模、易获取的 SIC 产品作为监督信号，对网络进行 SIC 任务预训练，使模型学习稳定的大尺度海冰分布表征。

2. **Stage 2：SIC-IW 联合训练**

   使用 Stage 1 得到的权重初始化模型，并在少量人工标注 IW 标签和对应 SIC 标签的监督下进行联合训练，以提升标签稀缺条件下的 IW 制图性能。

网络结构如下：
<img src="readme_image\SLAP-HiFNet.png" alt="SLAP-HiFNet"  />

## 4. 环境依赖

本项目代码在 Python 3.8、PyTorch 2.1.0 和 CUDA 11.8 环境下测试通过。建议使用 Conda 创建独立环境，并根据本机 CUDA 版本安装对应的 PyTorch。

主要依赖如下：

```txt
# Core deep learning environment
# Recommended Python version: 3.8
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.1.0+cu118
torchvision==0.16.0+cu118

# Numerical computing
numpy==1.24.3
scipy==1.10.1
pandas==2.0.3
scikit-learn==1.3.0
scikit-image==0.21.0

# Remote sensing and geospatial IO
rasterio==1.3.10
xarray==2023.1.0
netCDF4==1.6.2
tifffile==2023.7.10

# Image processing and visualization
opencv-python==4.10.0.84
Pillow==10.4.0
imageio==2.35.1
matplotlib==3.7.5

# Model backbones and segmentation models
timm==0.9.2
segmentation-models-pytorch==0.3.3
efficientnet-pytorch==0.7.1
pretrainedmodels==0.7.4

# Transformer / model utilities
einops==0.8.0
fvcore==0.1.5.post20221221
yacs==0.1.6
thop==0.1.1.post2209072238

# MMCV-related dependencies, used by some transformer backbones
mmcv==2.2.0
mmengine==0.10.7

# General utilities
tqdm==4.65.2
PyYAML==6.0.2
requests==2.28.2
rich==13.4.2
termcolor==2.4.0
addict==2.4.0
```

## 5. 数据准备

TODO：本项目数据集来自欧空局Sentinel-1和不来梅大学日级格网化AMSR2产品，标签来自不来梅MODIS-AMSR2，IW标签来自半自动化生成，正在整理中，将在[SLAP-HiFNet数据生成项目](https://github.com/mukysa/SLAP-HiFNet-DataPrep)中对生成流程开源。

## 6. 训练

训练入口为：

```bash
python train.py
```

在运行前，需要在 `train.py` 中设置模型名称、训练阶段和可选的 checkpoint 路径，例如：

```python
model_name = 'slapnet'
train_mode = 'stage1'
resume_ckpt_path = ''
```

进行 Stage 1 SIC 预训练时，设置：

```python
train_mode = 'stage1'
```

进行 Stage-linked SIC-IW 联合训练时，设置：

```python
train_mode = 'both'
```

在 'both' 模式下，代码会自动读取Stage 1的权重并在Stage 2加载。

如果需要加载已有 Stage 1 权重直接训练 Stage 2，设置：

```python
train_mode = 'stage2_resume'
resume_ckpt_path = 'xxxx\model_weights_file'
```

请将 `resume_ckpt_path` 设置为对应 checkpoint 的路径。

如果想要尝试

```python
train_mode = 'stage1'
```

或

```python
train_mode = 'stage2'
```

不同模型的训练参数位于：

```text
options/
```

其中，SLAP-HiFNet 的配置文件为：

```text
options/slapnet.py
```

**注意：**Stage 3是使用伪标签训练，用于本文推理结果的可靠性测试，我暂时没有将其相关代码删除，仅作论文复现参考时，请您忽略相关代码。

## 7. 测试与评价

测试入口为：

```bash
python test.py
```

在运行前，需要在 `test.py` 中设置模型名称、模型权重路径以及测试功能开关，例如：

```python
model_name = 'slapnet'
model_path = ''
model_type = 'last_best_model'

inference = True
cal_score = True
visual = True
```

其中：

- `inference` 控制是否执行推理；
- `cal_score` 控制是否计算评价指标；
- `visual` 控制是否生成可视化结果。

测试脚本支持完整场景的滑动窗口推理、评价指标计算和结果可视化。（三段式的代码主要出于GPU与CPU任务分流，节约成本，资源充足直接全部设为True即可）

脚本支持的可视化示例：

<img src="readme_image\S1A_EW_GRDM_1SDH_20250423T073932_20250423T074032_058881_074C66_2E19.nc_vis.png" alt="S1A_EW_GRDM_1SDH_20250423T073932_20250423T074032_058881_074C66_2E19.nc_vis" style="zoom:80%;" />

论文中的可视化示例：

<img src="readme_image\S1A_EW_GRDM_1SDH_20241024T164958_20241024T165058_056247_06E305_1964_matrix.png" alt="S1A_EW_GRDM_1SDH_20241024T164958_20241024T165058_056247_06E305_1964_matrix" style="zoom:67%;" />

## 8. 支持的模型

本仓库包含 SLAP-HiFNet 以及多种对照模型，主要包括：

- 单数据源模型
  - Sentinel-1 单源模型（ResNet系列， Swin Transformer）；
  - AMSR2 单源模型（ResNet系列， Swin Transformer）；
- 输入级融合模型：
  - ResNet-based models；
  - U-Net；
  - U-Net++；
  - DeepLabV3+；
  - PSPNet；
  - MANet；
  - XNet；
  - Swin Transformer；
  - CSwin Transformer；
- 特征级融合模型：
  - 单尺度特征级融合模型（双ResNet，双Swin）；
  - SLAP-HiFNet。

对应的配置文件位于 `options/` 目录下。

## 9. 预训练权重与实验结果

预训练 backbone 权重、模型 checkpoint 和实验输出结果默认不包含在本仓库中。建议将相关文件放置在用户自定义目录中，例如：

```text
pretrained_weights/
model_results/
```

权重获取方式：

- Transformer-本项目使用了改编自微软官方Swin Transformer和CSWin Transformer存储库的骨干实现：

  - Swin Transformer: https://github.com/microsoft/Swin-Transformer

  - CSWin Transformer: https://github.com/microsoft/CSWin-Transformer

- CNN-使用了来自SMP的网络构建与训练权重：https://github.com/qubvel-org/segmentation_models.pytorch 
- 本文的预训练权重（Stage1 和 Stage 2）：（待上传）

一个典型的 `model_results/` 目录结构如下：

```text
model_results/
├── data_cache/
│   ├── {scene_name}_sar.npy
│   ├── {scene_name}_mask.npy
│   ├── {scene_name}_label_80_sic_chart.npy
│   └── {scene_name}_label_manual_chart.npy
│
└── {model_name}_{train_mode}_{YYYY-MM-DD_HH-MM}/
    ├── s1_pretrain/
    │   ├── options.md
    │   ├── best_model
    │   ├── last_model
    │   ├── last_best_model
    │   ├── test_results/
    │   │   └── {model_type}/
    │   │       ├── {scene_name}_label_80_sic_output.npy
    │   │       └── ...
    │   └── visualization/
    │       └── {model_type}/
    │           ├── {scene_name}_vis.png
    │           └── per_scene_metrics.csv
    │
    ├── manual_finetune/
    │   ├── options.md
    │   ├── best_model
    │   ├── last_model
    │   ├── last_best_model
    │   ├── test_results/
    │   │   └── {model_type}/
    │   │       ├── {scene_name}_label_80_sic_output.npy
    │   │       ├── {scene_name}_label_manual_output.npy
    │   │       └── ...
    │   └── visualization/
    │       └── {model_type}/
    │           ├── {scene_name}_vis.png
    │           └── per_scene_metrics.csv
    │
    ├── pseudo_training/
    │   ├── options.md
    │   ├── best_model
    │   ├── last_model
    │   ├── last_best_model
    │   ├── test_results/
    │   │   └── {model_type}/
    │   │       ├── {scene_name}_label_80_sic_output.npy
    │   │       ├── {scene_name}_label_manual_output.npy
    │   │       └── ...
    │   └── visualization/
    │       └── {model_type}/
    │           ├── {scene_name}_vis.png
    │           └── per_scene_metrics.csv
    │
    ├── st2_on_st1/
    │   └── test_results/
    │       └── {model_type}/
    │           └── ...
    │
    ├── st1_on_st2/
    │   └── test_results/
    │       └── {model_type}/
    │           └── ...
    │
    └── manual_all/
        └── test_results/
            └── {model_type}/
                └── ...
```

其中：

- `s1_pretrain/` 表示 Stage 1 SIC 预训练结果。
- `manual_finetune/` 表示 Stage 2 SIC-IW 联合微调结果。
- `pseudo_training/` 表示 Stage 3 伪标签扩展训练结果。
- `st2_on_st1/` 表示使用 Stage 2 模型在 Stage 1 测试集上进行推理。
- `st1_on_st2/` 表示使用 Stage 1 模型在 Stage 2 测试集上进行推理。
- `manual_all/` 表示在完整人工标注数据集上进行推理，主要用于标签检查或后续分析。
- `options.md` 保存当前 stage 的完整配置，便于复现实验设置。
- `best_model` 保存验证集综合分数最高的模型。
- `last_model` 保存训练过程中按间隔保存的最近模型。
- `last_best_model` 保存训练后期验证集综合分数最高的模型，通常用于最终测试。
- `test_results/{model_type}/` 保存测试阶段输出的 `.npy` 预测结果。
- `visualization/{model_type}/` 保存可视化结果和逐场景指标统计。
- `data_cache/` 保存不同测试结果共享的 SAR、mask 和 GT label 缓存，避免在不同测试模式下重复保存相同数据。

## 10. 引用

如果本仓库对你的研究有帮助，请引用我们的论文。正式出版信息将后续更新。

```bibtex
@article{yang2026slaphifnet,
  title={SLAP-HiFNet: A Stage-Linked Active-Passive Microwave Hierarchical Fusion Network with Label-Efficient Transfer Learning for Sea Ice Mapping},
  author={Yang, Yushi and Feng, Tiantian and Jiang, Peng and Zhang, Liwen and Liu, Xiaomin and Hu, Yuxuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```

## 11. 致谢

感谢 ESA 提供 Sentinel-1 SAR 数据。感谢 University of Bremen 公开 AMSR2 亮温数据和 SIC 产品。

本代码实现使用了 PyTorch 以及若干开源 Python 库。

## 12. 许可证

本项目采用 MIT License。
