# -*- coding: utf-8 -*-
# @Time    : 2025/1/17 15:08
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: plot_process.py
# @Project : pancreatitis_diagnosis
import matplotlib.pyplot as plt


def plot_metrics(epoch,
                 train_losses=None, valid_losses=None, test_losses=None,
                 train_accuracies=None, valid_accuracies=None, test_accuracies=None,
                 learning_rates=None, save_path='./logs/training_process.png'
                 ):
    """
    绘制 Loss, Accuracy 和 Learning Rate 随 Epoch 的变化。
    Args:
        epoch (int): 当前 epoch 总数。
        train_losses (list): 每个 epoch 的训练损失。
        valid_losses (list): 每个 epoch 的验证损失。
        test_losses (list): 每个 epoch 的测试损失。
        train_accuracies (list): 每个 epoch 的训练准确率。
        valid_accuracies (list): 每个 epoch 的验证准确率。
        test_accuracies (list): 每个 epoch 的测试准确率。
        learning_rates (list, optional): 每个 epoch 的学习率。
        save_path (str, optional): 保存图片路径，默认为 None（不保存图片）。
    Returns:
        matplotlib.figure.Figure: 绘图对象。
    """
    fig, axs = plt.subplots(2 if learning_rates else 1, 1, figsize=(12, 10 if learning_rates else 6))

    # 绘制 Loss 曲线
    ax1 = axs[0] if learning_rates else axs
    ax1.plot(
        range(1, epoch + 1), train_losses, label='Train Loss', marker='o', color='blue'
    ) if train_losses else None
    ax1.plot(
        range(1, epoch + 1), valid_losses, label='Validation Loss', marker='o', color='orange'
    ) if valid_losses else None
    ax1.plot(
        range(1, epoch + 1), test_losses, label='Test Loss', marker='o', color='purple'
    ) if test_losses else None
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.grid()

    # 共享 x 轴，绘制 Accuracy 曲线
    ax2 = ax1.twinx()
    ax2.plot(
        range(1, epoch + 1), train_accuracies, label='Train Accuracy', marker='x', color='green'
    ) if train_losses else None
    ax2.plot(
        range(1, epoch + 1), valid_accuracies, label='Validation Accuracy', marker='x', color='red'
    ) if valid_losses else None
    ax2.plot(
        range(1, epoch + 1), test_accuracies, label='Test Accuracy', marker='x', color='cyan'
    ) if test_losses else None
    ax2.set_ylabel('Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    ax1.set_title('Loss and Accuracy vs. Epoch')

    # 绘制 Learning Rate 曲线
    if learning_rates:
        axs[1].plot(range(1, epoch + 1), learning_rates, label='Learning Rate', marker='o', color='purple')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Learning Rate')
        axs[1].set_title('Learning Rate vs. Epoch')
        axs[1].grid()
        axs[1].legend()

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    return fig
