import os
import json
from sklearn.utils import shuffle

# 所有nc文件所在目录
data_root = '/ssdfs/datahome/u10109/yushi/SASIC_MA'
json_save_path = '/share/home/u10109/data/yushi/CTSRNet/dataset/datalist_manual.json'

# 初始化两个列表
trainval_files = []
test_files = []

# 遍历所有文件
for file in os.listdir(data_root):
    if not file.endswith('.nc'):
        continue
    if '2024' in file:
        trainval_files.append(file)
    elif '2025' in file:
        test_files.append(file)

# 排序 + 打乱保证随机性 + 稳定性
trainval_files.sort()
trainval_files = shuffle(trainval_files, random_state=42)
test_files.sort()

# 确保可以均分为3份
assert len(trainval_files) >= 3, "训练验证样本不足以划分 3 fold"

# 手动分 3 fold
n_folds = 3
folds = [[] for _ in range(n_folds)]
for i, file in enumerate(trainval_files):
    folds[i % n_folds].append(file)

# 构造最终结构
result = {'folds': {}, 'test': test_files}

for i in range(n_folds):
    val_files = folds[i]
    train_files = [f for j, fold in enumerate(folds) if j != i for f in fold]
    result['folds'][f'fold{i}'] = {
        'train': sorted(train_files),
        'val': sorted(val_files)
    }

# 保存为 JSON 文件
with open(json_save_path, 'w') as f:
    json.dump(result, f, indent=4)

print(f"✅ 已保存交叉验证和测试列表：{json_save_path}")

