# -*- coding: utf-8 -*-
# @Time    : 2025/1/15 15:15
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: main.py
# @Project : pancreatitis_diagnosis
import torch
from torchvision import transforms
import logging
from models.resnet import *
from training.trianing import *
from training.lr_scheduler import *
from utils.augmentation import *
from utils.plot_process import *
from inference.inference import *
from datasets.pancreatitis_dataset import *


def setup_logging(log_file):
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,  # 日志级别
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 将日志写入文件
            # logging.StreamHandler()  # 在控制台打印日志
        ]
    )


def main():
    # 超参数设置
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 500
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    setup_logging('./logs/training.log')
    # 日志记录超参数
    logging.info("Starting Training Process")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Learning Rate: {learning_rate}")
    logging.info(f"Number of Epochs: {num_epochs}")

    # 示例：创建一个 ResNet 的 3D 模型
    # 模型保存地址(仅保存参数，使用 `model.load_state_dict(torch.load(best_model_path)))` 前需先获取模型结构
    best_model_path = './logs/3D_ResNet.pth'
    
    model = generate_model(10, n_input_channels=1, n_classes=2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: linear_lr(epoch, num_epochs)
    )

    # 加载训练和验证数据表
    train_df = pd.read_excel('../data/train.xlsx', index_col=0)
    valid_df = pd.read_excel('../data/validation.xlsx', index_col=0)
    test_df = pd.read_excel('../data/test.xlsx', index_col=0)
    infer_df = pd.read_excel('../data/train_validation_test.xlsx', index_col=0)

    # 定义数据变换
    transform = transforms.Compose([
        RandomCrop3D(crop_size=(30, 256, 256))
    ])

    # 构造训练和验证数据集
    train_dataset = PancreatitisDataset(
        dataframe=train_df,
        transform=transform,
        use_mask=False,
        use_mask_localization=False,
        set_window=False,
        window_width=400,
        window_level=50
    )

    valid_dataset = PancreatitisDataset(
        dataframe=valid_df,
        transform=transform,
        use_mask=False,
        use_mask_localization=False,
        set_window=False,
        window_width=400,
        window_level=50
    )

    test_dataset = PancreatitisDataset(
        dataframe=test_df,
        transform=transform,
        use_mask=False,
        use_mask_localization=False,
        set_window=False,
        window_width=400,
        window_level=50
    )

    infer_dataset = PancreatitisDataset(
        dataframe=infer_df,
        transform=transform,
        use_mask=False,
        use_mask_localization=False,
        set_window=False,
        window_width=400,
        window_level=50
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 设置 batch_size
        shuffle=True,  # 打乱数据
        num_workers=4,  # 使用多线程加载数据
        pin_memory=True  # 加速数据传输到 GPU（如果使用 GPU）
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,  # 验证时也可以调整 batch_size
        shuffle=False,  # 验证集无需打乱
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    best_score = 0.0
    train_losses, valid_losses, test_losses = [], [], []
    train_accuracies, valid_accuracies, test_accuracies = [], [], []
    learning_rates = []

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        train_loss, train_acc = train_epoch(model, criterion, train_loader, optimizer, device)
        valid_loss, valid_acc = validation_epoch(model, criterion, valid_loader, device)
        test_loss, test_acc = validation_epoch(model, criterion, test_loader, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)

        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
        test_accuracies.append(test_acc)

        # 记录日志
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - "
                     f"LR [{current_lr:.6f}] - "
                     f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                     f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f} - "
                     f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        scheduler.step()

        combined_score = train_acc + valid_acc + test_acc
        if combined_score > best_score and epoch >= 300:
            best_score = combined_score
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with score: {best_score:.4f}")

            # 推理并保存结果
            output_excel_path = f'./logs/inference_score{epoch+1}.xlsx'
            inference_and_save(model, infer_loader, device, output_excel_path)

        plot_metrics(
            epoch + 1, 
            train_losses, valid_losses, test_losses, 
            train_accuracies, valid_accuracies, test_accuracies, 
            learning_rates
        )

    logging.info("Training Process Completed")


if __name__ == "__main__":
    main()
