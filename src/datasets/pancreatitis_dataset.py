# -*- coding: utf-8 -*-
# @Time    : 2025/1/15 16:35
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: pancreatitis_dataset.py
# @Project : pancreatitis_diagnosis
import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader


class PancreatitisDataset(Dataset):
    def __init__(self,
                 dataframe,
                 transform=None,
                 use_mask=True,
                 use_mask_localization=False,
                 set_window=False,
                 window_width=500,
                 window_level=50,
                 debug_save=False):
        """初始化Dataset类。
        :param dataframe: 数据表以pandas.DataFrame方式保存
        :param transform: 数据增光
        :param use_mask: 是否使用仅胰腺区域
        :param use_mask_localization: 是否根据mask计算box中心裁剪
        :param set_window: 是否设置窗宽窗位
        :param window_width: 窗宽
        :param window_level: 窗位
        """
        self.data = dataframe
        self.transform = transform
        self.use_mask = use_mask
        self.use_mask_localization = use_mask_localization
        self.set_window = set_window
        self.window_width = window_width
        self.window_level = window_level
        self.debug_save = debug_save

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取索引
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径和标签
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = row['state']

        # 读取 NIfTI 图像
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image).astype('float32')
        mask_path = image_path.replace('images', 'labels')
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask).astype('float32')

        crop_size = (30, 150, 250)  # 原始范围，框定精确，经验值
        if self.use_mask_localization:
            # 找到 mask 的边界框
            non_zero_indices = np.argwhere(mask_array > 0)
            min_coords = non_zero_indices.min(axis=0)
            max_coords = non_zero_indices.max(axis=0)
            # 计算边界框的中心
            center_coords = ((min_coords + max_coords) / 2).astype(int)
            # 计算裁剪框的起始和结束坐标
            crop_min = np.maximum(center_coords - np.array(crop_size) // 2, 0)
            crop_max = np.minimum(center_coords + np.array(crop_size) // 2, image_array.shape)
            # 裁剪图像和掩模
            cropped_image = image_array[crop_min[0]:crop_max[0], crop_min[1]:crop_max[1], crop_min[2]:crop_max[2]]
            cropped_mask = mask_array[crop_min[0]:crop_max[0], crop_min[1]:crop_max[1], crop_min[2]:crop_max[2]]
        else:
            cropped_image, cropped_mask = image_array, mask_array

        combined_array = cropped_image * cropped_mask if self.use_mask else cropped_image  # 两种模式控制输入影像
        if self.set_window:
            min_val = self.window_level - self.window_width / 2
            max_val = self.window_level + self.window_width / 2
            combined_array = np.clip(combined_array, min_val, max_val)
        combined_array = self._resize_to_target(
            combined_array, crop_size
        ) if self.use_mask_localization else combined_array
        combined_array = np.expand_dims(combined_array, axis=0)  # 添加通道维度，形状变为 (1, D, H, W)

        if self.debug_save:
            debug_save_path = os.path.join('/home/huangdn/pancreatitis_diagnosis/data', os.path.basename(image_path))
            sitk_image = sitk.GetImageFromArray(combined_array)
            sitk.WriteImage(sitk_image, debug_save_path)

        x, y = torch.tensor(combined_array), torch.tensor(label, dtype=torch.long)
        x = self.transform(x) if self.transform else x

        return x, y

    @staticmethod
    def _resize_to_target(array, target_shape):
        """将裁剪后的数组调整为固定大小"""
        current_shape = array.shape
        padding = [(0, max(0, t - c)) for c, t in zip(current_shape, target_shape)]  # 计算每个维度的填充量
        padded_array = np.pad(array, padding, mode='constant', constant_values=0)
        slices = tuple(slice(0, t) for t in target_shape)
        return padded_array[slices]
