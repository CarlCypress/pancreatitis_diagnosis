# -*- coding: utf-8 -*-
# @Time    : 2025/1/12 14:32
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: resampling.py
# @Project : Radiological_image_processing
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def resample_z_direction(
        nii_path: str,
        is_mask: bool = False,
        output_file: str = 'output.nii.gz',
        spacing: float = 1.0,
        target_depth: int = None,
        is_print: bool = False
        ) -> int:
    """对 NIfTI 文件的 z 方向重采样，若 target_depth 不为 None，
    则根据 target_depth 变换厚度，否则根据物理距离 spacing.
    :param nii_path: 输入的 nifti 文件路径。
    :param is_mask: 是否是掩码文件?
    :param output_file: 重采样后保存的文件路径。
    :param spacing: z 轴的新体素间距(单位：mm)。
    :param target_depth: z轴采样目标厚度(单位：pixel)。
    :param is_print: 是否打印文件处理状态?
    :return: 原始影像z轴的厚度(方便逆操作)。
    """
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    affine = nii.affine
    z_spacing = np.abs(affine[2, 2])
    original_depth = data.shape[2]

    if target_depth is not None:
        # Calculate the zoom factor to match the target depth
        zoom_factor = target_depth / original_depth
        spacing = z_spacing / zoom_factor  # Update spacing based on the target depth
    else:
        # Use the provided physical spacing
        zoom_factor = z_spacing / spacing

    scale_factors = [1, 1, zoom_factor]
    new_data = zoom(data, scale_factors, order=0 if is_mask else 1)  # 对image数据采用线性插值，mask数据采用最近邻插值
    new_data = np.rint(new_data).astype(np.uint8) if is_mask else new_data  # mask 可能存在差值后的精度问题，需要舍入

    new_affine = affine.copy()
    new_affine[2, 2] = np.sign(affine[2, 2]) * spacing  # Update the z-spacing in the affine matrix

    new_img = nib.Nifti1Image(new_data, affine=new_affine, header=nii.header)
    nib.save(new_img, output_file)

    print(f'resampling {nii_path} completed.') if is_print else None
    return original_depth


