# -*- coding: utf-8 -*-
# @Time    : 2025/1/18 21:25
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: augmentation.py
# @Project : pancreatitis_diagnosis
import torch
import numpy as np


class RandomCrop3D:
    def __init__(self, crop_size):
        """初始化 RandomCrop3D。
        Args:
            crop_size (tuple): 裁剪目标大小 (depth, height, width)。
        """
        self.crop_size = crop_size

    def __call__(self, image):
        """对输入 3D 图像进行随机裁剪。
        Args:
            image (torch.Tensor): 输入 3D 图像，形状为 (C, D, H, W)。
        Returns:
            torch.Tensor: 裁剪后的 3D 图像，形状为 (C, crop_depth, crop_height, crop_width)。
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(image)}.")
        
        if image.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (C, D, H, W), but got {image.shape}.")

        _, depth, height, width = image.shape
        crop_depth, crop_height, crop_width = self.crop_size

        # 检查裁剪大小是否合理
        if crop_depth > depth or crop_height > height or crop_width > width:
            raise ValueError(
                f"Crop size {self.crop_size} exceeds image dimensions {image.shape[1:]}."
            )
        
        # 随机生成裁剪起始位置
        start_d = np.random.randint(0, depth - crop_depth + 1)
        start_h = np.random.randint(0, height - crop_height + 1)
        start_w = np.random.randint(0, width - crop_width + 1)
        
        # 裁剪图像
        cropped_image = image[
            :,  # 通道维度保持不变
            start_d:start_d + crop_depth,
            start_h:start_h + crop_height,
            start_w:start_w + crop_width
        ]

        return cropped_image


