# -*- coding: utf-8 -*-
"""用于划分训练验证集的函数集合"""
import json
import os
from sklearn.model_selection import train_test_split

# 读取 JSON 文件
def load_nc_data(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

# 随机划分训练集和验证集
def split_randomly(json_file_path, split_ratio=None, val_count=None, random_seed=None, exclude_files=None):
    nc_data = load_nc_data(json_file_path)
    
    # 构造所有文件的列表 (年份 + 文件名)
    all_files = [f"{year}/{file}" for year, files in nc_data.items() for file in files]
    
    # 如果提供了排除文件列表，则从 all_files 中移除这些文件
    if exclude_files:
        # 确保排除的文件是从 JSON 文件中加载的
        exclude_files = load_nc_data(exclude_files)  # 加载排除文件
        exclude_files = [f"{year}/{file}" for year, files in exclude_files.items() for file in files]
        files_to_split = [file for file in all_files if file not in exclude_files]
    else:
        files_to_split = all_files

    # 特殊情况：验证集为 0
    if val_count == 0 or (split_ratio is not None and split_ratio == 0):
        # 在每个项目前加上 "/test/"
        all_files = ["test/" + file for file in all_files]
        return all_files, []
    
    # 随机划分
    if val_count:  # 按数量划分
        train_files, val_files = train_test_split(files_to_split, test_size=val_count, random_state=random_seed)
    elif split_ratio:  # 按比例划分
        train_files, val_files = train_test_split(files_to_split, test_size=split_ratio, random_state=random_seed)
    else:
        raise ValueError("请提供 split_ratio 或 val_count 之一")

    # 将排除的文件加回到训练集中
    if exclude_files:
        train_files.extend(exclude_files)

    return train_files, val_files

# 按年份划分训练集和验证集
def split_by_year(json_file_path, train_years, val_years):
    nc_data = load_nc_data(json_file_path)

    train_files = [f"{year}/{file}" for year in train_years if year in nc_data for file in nc_data[year]]
    val_files = [f"{year}/{file}" for year in val_years if year in nc_data for file in nc_data[year]]

    return train_files, val_files


def get_s2_dataset_lists(json_path, options):
    data = load_nc_data(json_path)

    # 默认只使用 train 文件夹
    train_folders = options.get("trainset_options", ["train"])
    train_list = []
    for folder in train_folders:
        if folder in data:
            train_list.extend([f"{folder}/{fname}" for fname in data[folder]])

    # 验证和测试集直接固定读取
    val_list = [f"val/{fname}" for fname in data.get("val", [])]
    test_list = [f"test/{fname}" for fname in data.get("test", [])]

    return train_list, val_list, test_list


def get_manual_dataset_lists(json_file, fold_id=0):
    """
    读取手动标注数据集的训练、验证和测试列表。

    Args:
        json_file (str): JSON 文件路径
        fold_id (int): 当前 fold 的编号（0~2）

    Returns:
        train_list, val_list, test_list: 各子集文件名列表
    """
    data = load_nc_data(json_file)

    fold_key = f"fold{fold_id}"
    folds = data.get("folds", {})
    if fold_key not in folds:
        raise ValueError(f"{fold_key} 不存在于 JSON 文件中")

    train_list = folds[fold_key]["train"]
    val_list = folds[fold_key]["val"]
    test_list = data.get("test", [])

    return train_list, val_list, test_list

def get_manual_val_lists(json_file):
    """
    将 fold1 中的 train + val 全部作为 val_list 返回，用于全量验证。

    Args:
        json_file (str): JSON 文件路径

    Returns:
        train_list (empty), val_list (fold1 全部), test_list (原 test)
    """
    data = load_nc_data(json_file)
    folds = data.get("folds", {})
    if "fold1" not in folds:
        raise ValueError("fold1 不存在于 JSON 文件中")
    fold1 = folds["fold1"]
    val_list = fold1["train"] + fold1["val"]

    return val_list


def get_manual_list_all(json_file):
    """
    获取人工标注数据集的全部场景：fold0 的 train + val，加上 test 列表。

    Args:
        json_file (str): JSON 文件路径

    Returns:
        List[str]: 完整文件名列表（如：S1A_*.nc）
    """
    data = load_nc_data(json_file)

    # 获取 fold0 中的 train 和 val
    fold0 = data.get("folds", {}).get("fold0", {})
    fold0_train = fold0.get("train", [])
    fold0_val = fold0.get("val", [])

    # 获取 test
    test_list = data.get("test", [])

    # 合并并去重（理论上不重复）
    all_list = fold0_train + fold0_val + test_list

    return all_list


def get_combined_train_test_list(train_json_path, test_json_path, split_ratio=None, val_count=None, random_seed=None, exclude_files=None):
    # 获取训练+验证部分
    train_list, val_list = split_randomly(
        train_json_path,
        split_ratio=split_ratio,
        val_count=val_count,
        random_seed=random_seed,
        exclude_files=exclude_files
    )

    # 获取测试部分（val_count=0 时，split_randomly 会返回加前缀的）
    test_list, _ = split_randomly(test_json_path, val_count=0)

    # 合并并返回
    return train_list + val_list + test_list



# 示例使用
if __name__ == '__main__':
    json_file = "/share/home/u10109/data/yushi/CTSRNet/dataset/datalist.json"
    exclude_files = "/share/home/u10109/data/yushi/CTSRNet/dataset/excludelist.json"
    test_json_file = "/share/home/u10109/data/yushi/CTSRNet/dataset/testlist.json"

    # 示例 1：按比例随机划分 (80% 训练, 20% 验证)
    train_list, val_list = split_randomly(json_file, split_ratio=0.03, exclude_files=exclude_files)
    print(f"随机划分 - 训练集: {len(train_list)}, 验证集: {len(val_list)}")
    # print(f"训练集示例: {train_list[:5]}")
    # print(f"验证集示例: {val_list[:5]}")

    # 示例 2：按固定数量划分 (100 个验证集)
    train_list, val_list = split_randomly(json_file, val_count=25, exclude_files=exclude_files)
    print(f"随机划分 - 训练集: {len(train_list)}, 验证集: {len(val_list)}")

    # 示例 3：按年份划分 (2021 训练, 2022 验证)
    train_list, val_list = split_by_year(json_file, train_years=["2017", "2018", "2019", "2020","2021", "2022", "2023"], val_years=["2024"])
    print(f"按年份划分1 - 训练集: {len(train_list)}, 验证集: {len(val_list)}")
    
    train_list, val_list = split_by_year(json_file, train_years=["2017", "2018", "2019", "2020","2021", "2022", "2023", "2024"], val_years=[])
    print(f"按年份划分2 - 训练集: {len(train_list)}, 验证集: {len(val_list)}")

    # 示例 4：验证集为 0 的情况(测试集)
    train_list, val_list = split_randomly(test_json_file, val_count=0)
    print(f"随机划分 - 训练集: {len(train_list)}, 验证集: {len(val_list)}")
    # print(f"训练集示例: {train_list[:5]}")
    # print(f"验证集示例: {val_list}")
    
    # -- 读取Sentinel-2数据集
    s2_json_file = "/share/home/u10109/data/yushi/CTSRNet/dataset/datalist_s2.json"
    options = {
        # "trainset_options": ["train", "cloud", "move"]
        "trainset_options": ["train"]
    }

    train_list, val_list, test_list = get_s2_dataset_lists(s2_json_file, options)

    print(f"训练集: {len(train_list)} 条")
    print(f"验证集: {len(val_list)} 条")
    print(f"测试集: {len(test_list)} 条\n")
    
    # print(f"训练集示例: {train_list[:5]}")
    # print(f"验证集示例: {val_list}")

    # -- 读取manual数据集
    # manual_json_file = '/share/home/u10109/data/yushi/CTSRNet/dataset/datalist_manual.json'
    
    manual_json_file = '/share/home/u10109/data/yushi/CTSRNet/dataset/datalist_ma_cis.json'

    for fold_id in range(3):
        train_list, val_list, test_list = get_manual_dataset_lists(manual_json_file, fold_id)

        print(f"[Fold {fold_id}]")
        print(f"训练集数量: {len(train_list)}")
        print(f"验证集数量: {len(val_list)}")
        print(f"测试集数量: {len(test_list)}")

        # 可选：打印前几项检查
        # print(f"训练样例: {train_list[:2]}")
        # print(f"验证样例: {val_list}")
        
    # manual 验证集（stage3）
    val_list = get_manual_val_lists(manual_json_file)
    print(f"验证样本数: {len(val_list)}")
      
    # 全部manual数据
    full_list = get_manual_list_all(manual_json_file)

    print(f"人工标注数据集总数: {len(full_list)}")
    # for item in full_list:
    #     print(item)

    # train_json = "/share/home/u10109/data/yushi/CTSRNet/dataset/datalist.json"
    # test_json = "/share/home/u10109/data/yushi/CTSRNet/dataset/testlist.json"

    all_files = get_combined_train_test_list(
        train_json_path=json_file,
        test_json_path=test_json_file,
        val_count=20
    )

    print(f"总推理样本数: {len(all_files)}")
    # print(all_files)