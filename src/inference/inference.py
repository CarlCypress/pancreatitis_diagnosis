# -*- coding: utf-8 -*-
# @Time    : 2025/1/21 09:58
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: inference.py
# @Project : pancreatitis_diagnosis
import torch
import pandas as pd


def inference_and_save(model, data_loader, device, output_excel_path):
    """对指定数据集进行推理并保存结果"""
    model.eval()
    true_labels = []
    predicted_labels = []
    class_0_logits = []
    class_1_logits = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # 实际标签

            outputs = model(inputs)  # 模型输出 logits
            _, preds = torch.max(outputs, dim=1)  # 获取预测类别

            # 保存结果
            true_labels.append(labels.item())
            predicted_labels.append(preds.item())
            class_0_logits.append(outputs[0, 0].item())  # 保存 class 0 的 logits
            class_1_logits.append(outputs[0, 1].item())  # 保存 class 1 的 logits

    # 保存为 Excel 文件
    results_df = pd.DataFrame({
        "True_Label": true_labels,
        "Predicted_Label": predicted_labels,
        "Class_0_Logit": class_0_logits,
        "Class_1_Logit": class_1_logits
    })
    results_df.to_excel(output_excel_path, index=False)
    print(f"Inference results saved to {output_excel_path}")



# def inference_and_save(model, data_loader, device, output_excel_path):
#     """对指定数据集进行推理并保存结果"""
#     model.eval()
#     true_labels = []
#     predicted_labels = []
#     class_0_probs = []
#     class_1_probs = []

#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)  # 实际标签

#             outputs = model(inputs)  # 模型输出 logits
#             probs = torch.softmax(outputs, dim=1)  # 计算概率
#             _, preds = torch.max(outputs, dim=1)  # 获取预测类别

#             # 保存结果
#             true_labels.append(labels.item())
#             predicted_labels.append(preds.item())
#             class_0_probs.append(probs[0, 0].item())
#             class_1_probs.append(probs[0, 1].item())

#     # 保存为 Excel 文件
#     results_df = pd.DataFrame({
#         "True_Label": true_labels,
#         "Predicted_Label": predicted_labels,
#         "Class_0_Prob": class_0_probs,
#         "Class_1_Prob": class_1_probs
#     })
#     results_df.to_excel(output_excel_path, index=False)
#     print(f"Inference results saved to {output_excel_path}")
