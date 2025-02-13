# -*- coding: utf-8 -*-
# @Time    : 2025/1/16 20:05
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: trianing.py
# @Project : pancreatitis_diagnosis
import os
import time
import torch
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
import SimpleITK as sitk
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def train_epoch(model, criterion, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # print(f"Batch {batch_idx}: images.shape={images.shape}, labels.shape={labels.shape}")
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # print(f"Batch {batch_idx}: outputs={outputs}, loss={loss.item()}")
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # print(f"Batch {batch_idx}: predicted={predicted}, labels={labels}")

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validation_epoch(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # images, labels = images.unsqueeze(1).to(device), labels.to(device)
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

