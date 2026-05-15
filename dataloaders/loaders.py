# -*- coding: utf-8 -*-
"""dataloader, 针对多模态数据集的加载器(单输入端)"""

# --内置模块--
import os

# --第三方模块--
import numpy as np
import torch
import xarray as xr

# --自定义模块--

class SarAmsrTrainDataset(torch.utils.data.Dataset):
    """
    Multi-modal training dataset dataloader.
    多模态训练数据集的加载器。
    """

    def __init__(self, options):
        self.options = options
        self.files = self.options['train_list']
        self.sar_resolution = self.options['sar_resolution']
        self.label = 'label_80_sic' if self.sar_resolution == 80 else 'label_40_sic'  # 注意只是定位作用，变量获取与options['labels']有关

        # 补丁中的HR通道数量, 包括SAR通道数、上采样AMSR2通道数和参考冰图通道数
        self.patch_c_hr = len(self.options['sar_variables']) + 1 + len(self.options['amsr_variables']) + len(self.options['labels']) 

    def __len__(self):
        return self.options['epoch_len']
    
    def random_crop(self, scene):
        # 全分patch大小(高分+低分上采样)
        hr_patch = np.zeros((self.patch_c_hr, self.options['sar_patch_size'], self.options['sar_patch_size']))

        # 获取随机裁剪的范围
        row_rand = np.random.randint(low=0, high=scene[self.label].values.shape[0] - self.options['sar_patch_size'])
        col_rand = np.random.randint(low=0, high=scene[self.label].values.shape[1] - self.options['sar_patch_size'])
        
        # 将裁剪像素位置转换到整数倍位置上，以方便与低分辨率AMSR2数据对齐
        row_rand = int(row_rand // int(self.options['lr_delta']) * int(self.options['lr_delta']))
        col_rand = int(col_rand // int(self.options['lr_delta']) * int(self.options['lr_delta']))
        # 索引高分裁剪点对应的低分裁剪点
        lr_row = row_rand / self.options['lr_delta']
        lr_col = col_rand / self.options['lr_delta']

        # - Discard patches with too many meaningless pixels (optional).
        # - 丢弃包含太多无意义像素的补丁（可选）
        # 统计小于特定值的像素数(有效像素数量)
        less_than_value_count = np.sum(scene[self.label].values[row_rand: row_rand + self.options['sar_patch_size'],
                                                                col_rand: col_rand + self.options['sar_patch_size']] < self.options['ignore_values'])
        # 检查有效像素数量是否大于指定的数量，如果小于则丢弃该补丁，大于则开始裁剪
        if less_than_value_count > self.options['min_pixels']:
            # --裁剪 full(SAR+label) 变量--
            if self.sar_resolution == 80:  # 高分为80m
                sar_patch = scene[self.options['full_variables']].isel(
                    y_80=range(row_rand, row_rand + self.options['sar_patch_size']),
                    x_80=range(col_rand, col_rand + self.options['sar_patch_size'])).to_array().values
            elif self.sar_resolution == 40:  # 高分为40m
                sar_patch = scene[self.options['full_variables']].isel(
                    y_40=range(row_rand, row_rand + self.options['sar_patch_size']),
                    x_40=range(col_rand, col_rand + self.options['sar_patch_size'])).to_array().values
            # 计算极化对比差 (HH - HV)
            polarization_diff = sar_patch[len(self.options['labels'])] - sar_patch[len(self.options['labels'])+1]
            # 将极化对比差添加到高分补丁中
            hr_patch[0:len(self.options['full_variables']), :, :] = sar_patch
            hr_patch[len(self.options['full_variables']) , :, :] = polarization_diff
            
            # --裁剪并上采样amsr2变量--
            hr_patch[len(self.options['full_variables']) + 1:, :, :] = torch.nn.functional.interpolate(
                    input=torch.from_numpy(scene[self.options['amsr_variables']].to_array().values[
                                        :,
                                        int(lr_row): int(lr_row + np.ceil(self.options['amsr_patch_size'])),
                                        int(lr_col): int(lr_col + np.ceil(self.options['amsr_patch_size']))]
                                        ).unsqueeze(0),
                    size=(self.options['sar_patch_size'], self.options['sar_patch_size']),
                    mode=self.options['loader_upsampling']).squeeze(0)[
                                                                    :,
                                                                    :self.options['sar_patch_size'],
                                                                    :self.options['sar_patch_size']
                                                                        ].numpy()
            
        # In case patch does not contain any valid pixels - return None.
        # 如果补丁不包含任何有效像素，则返回 None。
        else:
            hr_patch = None
        
        return hr_patch

    def prep_dataset(self, chart_hr_patches):
        # 转为张量
        hr_tensor = torch.from_numpy(chart_hr_patches[:, len(self.options['labels']):]).type(torch.float)
        label_dict = {}
        for idx, label in enumerate(self.options['labels']):
            label_dict[label] = torch.from_numpy(chart_hr_patches[:, idx]).type(torch.long)

        return hr_tensor, label_dict

    def __getitem__(self, idx):
        # Placeholder to fill with data.
        # 占位符，用于填充数据。
        chart_hr_patches = np.zeros((self.options['batch_size'], self.patch_c_hr,
                            self.options['sar_patch_size'], self.options['sar_patch_size']))
        sample_n = 0

        # Continue until batch is full.
        # 继续直到batch填满为止
        while sample_n < self.options['batch_size']:
            # - Open memory location of scene. Uses 'Lazy Loading'.
            # 打开场景的内存位置。使用“Lazy Loading”
            scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()

            # - Load scene
            # 加载场景
            scene = xr.open_dataset(os.path.join(self.options['path_to_train_data'], self.files[scene_id]))
            # - 提取随机裁剪的补丁
            try:
                hr_scene_patch = self.random_crop(scene)
            except:
                print(f"Cropping in {self.files[scene_id]} failed.")
                print(
                    f"Scene size: {scene[self.label].values.shape} for crop shape: ({self.options['sar_patch_size']}, {self.options['sar_patch_size']})")
                print('Skipping scene.')
                continue
            # hr_scene_patch = self.random_crop(scene)  # 调试代码所需

            if hr_scene_patch is not None:
                # -- 将场景补丁堆叠在patch中
                chart_hr_patches[sample_n, :, :, :] = hr_scene_patch
                sample_n += 1  # Update the index.

        # Prepare training arrays
        # 准备训练数组
        hr_tensor, label_dict = self.prep_dataset(chart_hr_patches=chart_hr_patches)

        return hr_tensor, label_dict


class SarAmsrPatchValDataset(torch.utils.data.Dataset):
    """
    Multi-modal validation and test dataset dataloader.
    多模态验证和测试数据集的加载器。
    """
    def __init__(self, options, files):
        self.options = options
        self.files = files
        self.sar_resolution = self.options['sar_resolution']
        self.label = 'label_80_sic' if self.sar_resolution == 80 else 'label_40_sic'
        self.window_size = self.options['window_size']  # 滑动窗口大小
        self.stride = self.options['stride']  # 滑动步长
        self.batch_size = self.options['val_batch_size']

    def __len__(self):
        """
        Provide the number of iterations. Function required by Pytorch dataset.
        提供迭代次数。PyTorch dataset所需的函数

        Returns
        -------
        Number of scenes per validation.。
        验证中的场景数量。
        """
        return len(self.files)
    
    def prep_scene(self, scene):
        """
        Upsample low resolution to match charts and SAR resolution. Convert patches from 4D numpy array to 4D torch tensor.
        将低分辨率图像上采样以匹配图表和 SAR 分辨率。将补丁从 4D NumPy 数组转换为 4D Torch 张量

        Parameters
        ----------
        scene :
            Xarray dataset; a scene from SASIC dataset.
            来自 SASIC 数据集的一个场景。

        Returns
        -------
        cropped_hr_scene :
            4D torch tensor, ready training data.
        lr_scene :
            4D torch tensor, ready training data.
        chart :
            3D torch tensors; reference inference data for x. None if test is true.
            3D torch tensor；对于 x 的参考推断数据。如果 test 为真，则为 None。
        """
        # 高分数据
        sar_data = torch.from_numpy(scene[self.options['sar_variables']].to_array().values)
        # 计算极化对比差 (HH - HV)
        polarization_diff = sar_data[0] - sar_data[1]
        sar_data = torch.cat((sar_data, polarization_diff.unsqueeze(0)), dim=0)  # 将极化对比差添加到SAR数据后面
        # 低分数据
        amsr_data = torch.from_numpy(scene[self.options['amsr_variables']].to_array().values)
        
        # 上采样低分数据
        amsr_upsampled = torch.nn.functional.interpolate(
            input=amsr_data.unsqueeze(0),
            size=sar_data.shape[-2:],
            mode=self.options['loader_upsampling']
        ).squeeze(0)
        
        # 合并高分和上采样低分数据
        hr_scene = torch.cat((sar_data, amsr_upsampled), dim=0)  # 形状应该是 [channels, height, width]

        # 获取并裁剪label冰图
        chart = scene[self.label].values
        # 生成掩码
        mask = (chart > self.options['ignore_values'])

        # 无重叠地裁剪 patch
        hr_windows = []
        chart_windows = []
        mask_windows = []

        height, width = hr_scene.size(-2), hr_scene.size(-1)
        for i in range(0, height - self.window_size + 1, self.stride):
            for j in range(0, width - self.window_size + 1, self.stride):
                hr_window = hr_scene[:, i:i + self.window_size, j:j + self.window_size]  # 形状为 [channels, H, W]
                chart_window = chart[i:i + self.window_size, j:j + self.window_size]
                mask_window = mask[i:i + self.window_size, j:j + self.window_size]
                # 统计无效像素数量
                invalid_pixels = np.sum(mask_window)  # 使用 mask_window 统计

                # 检查无效像素数量是否小于阈值
                if invalid_pixels < self.options['min_pixels']:
                    hr_windows.append(hr_window)
                    chart_windows.append(chart_window)
                    mask_windows.append(mask_window)
        
        if self.batch_size > 0 and len(hr_windows) > self.batch_size:
            indexes = list(range(len(hr_windows)))
            selected_indexes = np.random.choice(indexes, size=self.batch_size, replace=False)
            # 根据索引筛选patch
            hr_windows = [hr_windows[i] for i in selected_indexes]
            chart_windows = [chart_windows[i] for i in selected_indexes]
            mask_windows = [mask_windows[i] for i in selected_indexes]

        return hr_windows, chart_windows, mask_windows

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.
        获取场景。PyTorch 数据集所需的函数。

        Returns
        -------
        hr_scene :
            4D torch tensor; ready inference data. 准备好的推断数据。
        lr_scene :
            4D torch tensor; ready inference data. 准备好的推断数据。
        chart :
            3D torch tensors; reference inference data for x. None if test is true.
            3D Torch 张量；对于 x 的参考数据。如果 test 为真，则为 None。
        masks :
            Dict with 2D torch tensors; mask for each chart for loss calculation. Contain only SAR mask if test is true.
            label的 2D Torch 张量字典；用于损失计算的掩模。如果 test 为真，则仅包含 SAR 掩模。
        name : str
            Name of scene.
        """
        scene = xr.open_dataset(os.path.join(self.options['path_to_test_val_data'], self.files[idx]))

        name = self.files[idx]
        hr_windows, chart_windows, mask_windows = self.prep_scene(scene)

        # 将 windows 转换为批量张量
        if hr_windows:
            # 合并 HR windows
            batch_hr = torch.stack(hr_windows, dim=0)
            
            # 处理 chart 和 mask windows
            batch_chart = torch.tensor(np.stack(chart_windows, axis=0), dtype=torch.long).unsqueeze(1)  # 增加通道维度
            batch_mask = torch.tensor(np.stack(mask_windows, axis=0), dtype=torch.bool).unsqueeze(1)  # 增加通道维度
        else:
            # 处理无窗口的情况，返回空张量或默认值
            batch_hr = torch.empty(0)
            batch_chart = torch.empty(0)
            batch_mask = torch.empty(0)

        return batch_hr, batch_chart, batch_mask, name


class SarAmsrSenceValTestDataset(torch.utils.data.Dataset):
    """
    Multi-modal test dataset dataloader.
    多模态测试数据集的加载器。
    """
    def __init__(self, options, files):
        self.options = options
        self.files = files
        self.sar_resolution = self.options['sar_resolution']
        self.label = 'label_80_sic' if self.sar_resolution == 80 else 'label_40_sic'
        self.window_size = self.options['test_window_size']  # 滑动窗口大小
        self.stride = self.options['test_stride']  # 滑动步长

    def __len__(self):
        return len(self.files)

    def pad_scene(self, scene, height, width):
        # 计算需要填充的大小
        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size

        # 填充场景
        padded_scene = torch.nn.functional.pad(scene, (0, pad_width, 0, pad_height), mode='reflect')

        return padded_scene, height + pad_height, width + pad_width

    def prep_scene(self, scene):
        # 高分数据
        sar_scene = torch.from_numpy(scene[self.options['sar_variables']].to_array().values)
        sar = sar_scene[:2]  # 获取第一张SAR数据
        # 计算极化对比差 (HH - HV)
        polarization_diff = sar_scene[0] - sar_scene[1]
        sar_scene = torch.cat((sar_scene, polarization_diff.unsqueeze(0)), dim=0)  # 将极化对比差添加到SAR数据后面
        
        # 低分数据
        amsr_scene = torch.from_numpy(scene[self.options['amsr_variables']].to_array().values)
        # 上采样低分数据
        amsr_upsampled = torch.nn.functional.interpolate(
            input=amsr_scene.unsqueeze(0),
            size=sar_scene.shape[-2:],
            mode=self.options['loader_upsampling']
        ).squeeze(0)
        
        # 合并高分和上采样低分数据
        hr_scene = torch.cat((sar_scene, amsr_upsampled), dim=0)  # 形状应该是 [channels, height, width]

        # 获取并裁剪label冰图
        label_dict = {}
        for label_name in self.options['labels']:
            if label_name in scene.variables:
                label_dict[label_name] = scene[label_name].values

        # 掩膜和尺寸统一使用主标签（如 label_80_sic）
        if 'label_80_sic' in scene.variables:
            main_label = 'label_80_sic'
        elif self.options['labels'][0] in scene.variables:
            main_label = self.options['labels'][0]
        else:
            raise KeyError(f"Neither 'label_80_sic' nor '{self.options['labels'][0]}' found in scene.variables")
        mask = (scene[main_label].values > self.options['ignore_values'])

        # 获取场景的原始形状
        original_height, original_width = scene[main_label].shape
        scene_shape = (original_height, original_width)

        # 填充场景
        padded_scene, new_height, new_width = self.pad_scene(hr_scene, original_height, original_width)
        pad_shape = (new_height, new_width)

        # 无重叠地裁剪 patch
        hr_windows = []
        window_positions = []

        height, width = padded_scene.size(-2), padded_scene.size(-1)
        for i in range(0, height - self.window_size + 1, self.stride):
            for j in range(0, width - self.window_size + 1, self.stride):
                hr_window = padded_scene[:, i:i + self.window_size, j:j + self.window_size]  # 形状为 [channels, H, W]
                hr_windows.append(hr_window)
                window_positions.append((i, j))

        return hr_windows, label_dict, mask, window_positions, scene_shape, pad_shape, sar

    def __getitem__(self, idx):
        scene = xr.open_dataset(os.path.join(self.options['path_to_test_val_data'], self.files[idx]))

        name = self.files[idx]
        hr_windows, label_dict, mask, window_positions, scene_shape, pad_shape, sar = self.prep_scene(scene)

        return hr_windows, label_dict, mask, window_positions, scene_shape, pad_shape, sar, name


def get_variable_options(train_options: dict):
    """
    Get amsr and env grid options, crop shape and upsampling shape.
    获取 amsr 和 env 网格选项，裁剪和上采样形状

    Parameters
    ----------
    train_options: dict
        Dictionary with training options.
    
    Returns
    -------
    train_options: dict
        Updated with lr options. 更新为包含 lr 选项的字典
    """
    # 采样倍数差，SASIC数据集中，40m与80m的SAR与AMSR2的采样倍数差分别为64和32
    train_options['lr_delta'] = 64 / (train_options['sar_resolution'] // 40)
    # AMSR2补丁大小(直接在预设阶段保证整除)
    train_options['lr_patch'] = train_options['sar_patch_size'] / train_options['lr_delta']
    
    # 将冰图和高分数据合并，将用于后续读取数据处理部分
    train_options['full_variables'] = np.hstack((train_options['labels'], train_options['sar_variables']))

    # 计算最低有效像素数量
    train_options['min_pixels'] = train_options['min_pixels_thd'] * train_options['sar_patch_size'] * train_options['sar_patch_size']

    return train_options
