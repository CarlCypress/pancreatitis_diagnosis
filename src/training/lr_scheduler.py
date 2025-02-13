# -*- coding: utf-8 -*-
# @Time    : 2025/1/18 19:12
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: lr_scheduler.py
# @Project : pancreatitis_diagnosis
# training/lr_scheduler.py
def linear_lr(epoch, total_epochs):
    """线性学习率衰减函数。
    Args:
        epoch (int): 当前的 epoch 数。
        total_epochs (int): 总的 epoch 数。
    Returns:
        float: 当前 epoch 对应的学习率缩放因子。
    """
    return 1 - epoch / total_epochs
