import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os


class PlotConfig:
    """绘图配置类 - 统一管理中英文字体和样式"""
    
    def __init__(self):
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['Songti SC', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 设置serif字体，用于正式报告
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Songti SC', 'SimSun', 'Times New Roman']
        
        # 图表样式
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 100
        
        # 网格和边距
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11
        
        # 颜色配置
        self.colors = {
            'lenet': '#2E86AB',      # 蓝色
            'vgg': '#A23B72',        # 紫色  
            'resnet': '#F18F01',     # 橙色
            'background': '#F5F5F5'   # 浅灰
        }
        
        # 确保输出目录存在
        os.makedirs('assets/figures', exist_ok=True)
    
    def get_color(self, model_name: str) -> str:
        """获取模型对应的颜色"""
        return self.colors.get(model_name.lower(), '#333333')


def plot_loss_curves(loss_histories: Dict[str, List[float]], 
                    title: str = "训练损失曲线对比",
                    save_path: Optional[str] = None) -> None:
    """
    绘制多个模型的损失曲线对比图
    
    Args:
        loss_histories: 字典，键为模型名，值为损失历史列表
        title: 图表标题
        save_path: 保存路径，如果为None则使用默认路径
    """
    config = PlotConfig()
    
    plt.figure(figsize=(12, 8))
    
    # 绘制每个模型的损失曲线
    for model_name, losses in loss_histories.items():
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, 
                label=model_name, 
                color=config.get_color(model_name.lower()),
                linewidth=2, 
                alpha=0.8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('训练轮次 (Epoch)', fontsize=14)
    plt.ylabel('损失值 (Loss)', fontsize=14)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # 设置y轴从0开始
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = 'assets/figures/loss_curves_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"损失曲线图已保存至: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, 
                         model_name: str = "Model",
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> None:
    """
    绘制混淆矩阵热力图
    
    Args:
        cm: 混淆矩阵 (num_classes, num_classes)
        model_name: 模型名称（用于文件名）
        title: 图表标题
        save_path: 保存路径
    """
    config = PlotConfig()
    
    # 类别标签（数字0-9）
    class_names = [str(i) for i in range(cm.shape[0])]
    
    if title is None:
        title = f'{model_name} 混淆矩阵'
    
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': '样本数量'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    
    # 调整标签旋转角度
    plt.xticks(rotation=0)  # 数字标签不需要旋转
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = f'assets/figures/confusion_matrix_{model_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵已保存至: {save_path}")


def plot_error_samples(images: np.ndarray, 
                       true_labels: np.ndarray, 
                       pred_labels: np.ndarray,
                       model_names: List[str],
                       title: str = "三个模型共同的错误样本",
                       save_path: Optional[str] = None) -> None:
    """
    绘制多个模型都预测错误的样本
    
    Args:
        images: 图像数据 (num_samples, 1, 28, 28)
        true_labels: 真实标签
        pred_labels: 各模型的预测标签 (num_models, num_samples)
        model_names: 模型名称列表
        title: 图表标题
        save_path: 保存路径
    """
    config = PlotConfig()
    
    # 找出所有模型都预测错误的样本
    num_models = len(model_names)
    all_wrong_mask = np.ones(len(images), dtype=bool)
    
    for i in range(num_models):
        all_wrong_mask &= (pred_labels[i] != true_labels)
    
    wrong_images = images[all_wrong_mask]
    wrong_true = true_labels[all_wrong_mask]
    wrong_preds = pred_labels[:, all_wrong_mask]
    
    num_samples = min(16, len(wrong_images))  # 最多显示16个样本
    
    if num_samples == 0:
        print("没有找到所有模型都预测错误的样本！")
        return
    
    # 创建网格布局
    rows = (num_samples + 3) // 4
    cols = min(4, num_samples)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i] if num_samples > 1 else axes
        
        # 显示图像
        img = wrong_images[i].squeeze()
        ax.imshow(img, cmap='gray', interpolation='nearest')
        
        # 构建标题
        true_label = wrong_true[i]
        preds_list = [f'{name}:{pred}' for name, pred in zip(model_names, wrong_preds[:, i])]
        preds_str = '\\n'.join(preds_list)
        
        ax.set_title(f'真实:{true_label}\\n{preds_str}', fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = 'assets/figures/common_error_samples.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"错误样本图已保存至: {save_path}")
    print(f"找到 {len(wrong_images)} 个所有模型都预测错误的样本")


def plot_accuracy_comparison(accuracy_metrics: Dict[str, Dict[str, float]], 
                         title: str = "模型准确率对比",
                         save_path: Optional[str] = None) -> None:
    """
    绘制模型准确率对比柱状图
    
    Args:
        accuracy_metrics: 准确率指标字典 {model_name: {'accuracy': value}}
        title: 图表标题
        save_path: 保存路径
    """
    config = PlotConfig()
    
    # 提取数据
    models = list(accuracy_metrics.keys())
    accuracies = []
    
    for model in models:
        # 兼容不同的键名
        if '准确率' in accuracy_metrics[model]:
            accuracies.append(accuracy_metrics[model]['准确率'])
        elif 'accuracy' in accuracy_metrics[model]:
            accuracies.append(accuracy_metrics[model]['accuracy'])
        else:
            accuracies.append(0.0)
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [config.get_color(model.lower()) for model in models]
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7)
    
    # 在柱状图上显示数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('模型', fontsize=14)
    ax.set_ylabel('准确率', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 设置y轴范围以突出差异
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    ax.set_ylim(min_acc - 0.01, max_acc + 0.01)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = 'assets/figures/accuracy_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"准确率对比图已保存至: {save_path}")


def plot_training_time_comparison(time_metrics: Dict[str, Dict[str, float]], 
                             title: str = "模型训练时间对比",
                             save_path: Optional[str] = None) -> None:
    """
    绘制模型训练时间对比柱状图
    
    Args:
        time_metrics: 时间指标字典 {model_name: {'time': value}}
        title: 图表标题
        save_path: 保存路径
    """
    config = PlotConfig()
    
    # 提取数据
    models = list(time_metrics.keys())
    times = []
    
    for model in models:
        # 兼容不同的键名
        if '训练时间' in time_metrics[model]:
            times.append(time_metrics[model]['训练时间'])
        elif '训练时间(s)' in time_metrics[model]:
            times.append(time_metrics[model]['训练时间(s)'])
        elif 'time' in time_metrics[model]:
            times.append(time_metrics[model]['time'])
        else:
            times.append(0.0)
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [config.get_color(model.lower()) for model in models]
    bars = ax.bar(models, times, color=colors, alpha=0.7)
    
    # 在柱状图上显示数值
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('模型', fontsize=14)
    ax.set_ylabel('训练时间 (秒)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = 'assets/figures/training_time_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练时间对比图已保存至: {save_path}")


def plot_model_comparison_table(results: Dict[str, Dict[str, float]], 
                           title: str = "模型性能总结",
                           save_path: Optional[str] = None) -> None:
    """
    绘制模型性能总结表格
    
    Args:
        results: 结果字典 {model_name: {metrics}}
        title: 图表标题
        save_path: 保存路径
    """
    config = PlotConfig()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    models = list(results.keys())
    metrics = ['准确率', '训练时间(s)']
    
    table_data = []
    for model in models:
        row = [model]
        model_results = results[model]
        
        for metric in metrics:
            if metric in model_results:
                if metric == '准确率':
                    row.append(f'{model_results[metric]:.4f}')
                else:
                    row.append(f'{model_results[metric]:.1f}')
            else:
                row.append('-')
        
        table_data.append(row)
    
    # 创建表格
    table = ax.table(cellText=table_data,
                   colLabels=['模型'] + metrics,
                   cellLoc='center',
                   loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # 设置表格样式
    table[(0, 0)].set_facecolor('#40466e')
    table[(0, 0)].set_text_props(weight='bold', color='white')
    
    for i in range(len(metrics)):
        table[(0, i+1)].set_facecolor('#40466e')
        table[(0, i+1)].set_text_props(weight='bold', color='white')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = 'assets/figures/model_summary_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"性能总结表格已保存至: {save_path}")


if __name__ == "__main__":
    # 测试绘图功能
    print("测试绘图配置...")
    
    # 测试损失曲线
    loss_histories = {
        'LeNet': [2.3, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6],
        'VGG': [2.1, 1.5, 1.1, 0.8, 0.6, 0.4, 0.3],
        'ResNet': [2.0, 1.3, 0.9, 0.6, 0.4, 0.2, 0.15]
    }
    plot_loss_curves(loss_histories, "测试损失曲线")
    
    # 测试混淆矩阵
    cm = np.random.randint(0, 100, (10, 10))
    np.fill_diagonal(cm, np.random.randint(800, 1000, 10))  # 对角线设为较大的值
    plot_confusion_matrix(cm, "LeNet5", "LeNet-5 混淆矩阵")
    
    # 测试准确率对比
    accuracy_metrics = {
        'LeNet-5': {'准确率': 0.985},
        'VGG': {'准确率': 0.992},
        'ResNet': {'准确率': 0.994}
    }
    plot_accuracy_comparison(accuracy_metrics, "测试准确率对比")
    
    # 测试训练时间对比
    time_metrics = {
        'LeNet-5': {'训练时间': 245.3},
        'VGG': {'训练时间': 89.7},
        'ResNet': {'训练时间': 127.8}
    }
    plot_training_time_comparison(time_metrics, "测试训练时间对比")
    
    print("绘图功能测试完成！")
