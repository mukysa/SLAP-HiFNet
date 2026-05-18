# -*- coding: utf-8 -*-
"""用于创建json, 标记文件位置"""
import os
import json

# 母文件夹路径
# parent_folder = '/cehui1/yushi/SASIC/'
parent_folder = '/cehui1/yushi/SASIC/test/'
# 指定存储json文件的文件夹路径
json_folder = '/cehui1/yushi/CTSRNet/dataset'

# 创建一个字典来存储结果
result = {}

# 遍历母文件夹下的每个以年份命名的文件夹
for year_folder in os.listdir(parent_folder):
    # 拼接完整的文件夹路径
    folder_path = os.path.join(parent_folder, year_folder)
    # 确保是文件夹
    if os.path.isdir(folder_path):
        # 初始化一个列表来存储该年份文件夹下的nc文件名
        nc_files = []
        # 遍历文件夹中的文件
        for file in os.listdir(folder_path):
            # 判断文件是否为nc文件
            if file.endswith('.nc'):
                nc_files.append(file)
        # 将该年份的nc文件名列表存储到字典中
        result[year_folder] = nc_files

# 拼接json文件的完整路径
json_file_path = os.path.join(json_folder, 'testlist.json')

# 将字典写入json文件
with open(json_file_path, 'w') as f:
    json.dump(result, f, indent=4)

print(f'json文件已成功存储到{json_file_path}')