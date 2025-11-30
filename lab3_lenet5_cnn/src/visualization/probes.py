import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
import os

# 导入PlotConfig
try:
    from .plotting import PlotConfig
except ImportError:
    # 如果作为独立脚本运行
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from visualization.plotting import PlotConfig


def visualize_conv_kernels(weights: np.ndarray, 
                          layer_name: str = "Conv1",
                          title: Optional[str] = None,
                          save_path: Optional[str] = None,
                          max_kernels: int = 16) -> None:
    """
    可视化单个卷积层的权重
    
    Args:
        weights: 卷积核权重 (out_channels, in_channels, kernel_h, kernel_w)
        layer_name: 层名称
        title: 图表标题
        save_path: 保存路径
        max_kernels: 最多显示的卷积核数量
    """
    """
    可视化卷积核权重
    
    Args:
        weights: 卷积核权重 (out_channels, in_channels, kernel_h, kernel_w)
        layer_name: 层名称
        title: 图表标题
        save_path: 保存路径
        max_kernels: 最多显示的卷积核数量
    """
    config = PlotConfig()
    
    out_channels, in_channels, kh, kw = weights.shape
    
    if title is None:
        title = f'{layer_name} 卷积核可视化'
    
    # 确定显示的卷积核数量
    num_display = min(max_kernels, out_channels)
    
    # 计算网格大小
    cols = min(4, num_display)
    rows = (num_display + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(num_display):
        ax = axes[i] if num_display > 1 else axes
        
        # 对于多通道输入，取平均值或显示第一个通道
        if in_channels == 1:
            kernel = weights[i, 0]  # 单通道
        else:
            # 多通道：显示RGB合成图或平均
            if in_channels == 3:
                kernel = np.transpose(weights[i], (1, 2, 0))  # HWC格式
                # 归一化到[0,1]
                kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
            else:
                kernel = np.mean(weights[i], axis=0)  # 平均所有通道
        
        # 显示卷积核
        if len(kernel.shape) == 2:  # 灰度图
            im = ax.imshow(kernel, cmap='viridis', interpolation='nearest')
        else:  # RGB图
            im = ax.imshow(kernel)
        
        ax.set_title(f'Kernel {i+1}', fontsize=10)
        ax.axis('off')
        
        # 添加colorbar (只对灰度图)
        if len(kernel.shape) == 2:
            plt.colorbar(im, ax=ax, shrink=0.6)
    
    # 隐藏多余的子图
    for i in range(num_display, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = f'assets/figures/{layer_name.lower()}_kernels.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"卷积核可视化已保存至: {save_path}")


def visualize_feature_maps(feature_maps: List[torch.Tensor],
                          input_image: torch.Tensor,
                          layer_names: Optional[List[str]] = None,
                          title: str = "ResNet 特征图可视化",
                          save_path: Optional[str] = None,
                          max_channels: int = 16) -> None:
    """
    可视化特征图
    
    Args:
        feature_maps: 各层的特征图列表
        input_image: 输入图像 (1, 1, 28, 28)
        layer_names: 各层名称列表
        title: 图表标题
        save_path: 保存路径
        max_channels: 每层最多显示的通道数
    """
    config = PlotConfig()
    
    if layer_names is None:
        layer_names = [f'Layer {i+1}' for i in range(len(feature_maps))]
    
    # 显示输入图像
    fig, axes = plt.subplots(len(feature_maps) + 1, max_channels + 1, 
                            figsize=(4 * (max_channels + 1), 3 * (len(feature_maps) + 1)))
    
    if len(feature_maps) == 0:
        return
    
    # 显示输入图像
    input_img = input_image.squeeze().cpu().numpy()
    axes[0, 0].imshow(input_img, cmap='gray')
    axes[0, 0].set_title('输入图像', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 隐藏输入图像行的其他列
    for j in range(1, max_channels + 1):
        axes[0, j].axis('off')
    
    # 显示各层的特征图
    for i, (feat_map, layer_name) in enumerate(zip(feature_maps, layer_names)):
        row = i + 1
        
        # 显示层名称
        axes[row, 0].text(0.5, 0.5, layer_name, 
                         ha='center', va='center', fontsize=12, 
                         fontweight='bold', transform=axes[row, 0].transAxes)
        axes[row, 0].axis('off')
        
        # 转换为numpy并取第一个样本
        feat_np = feat_map[0].cpu().numpy()  # (channels, H, W)
        num_channels = feat_np.shape[0]
        
        # 显示前几个通道的特征图
        num_display = min(max_channels, num_channels)
        
        for j in range(num_display):
            col = j + 1
            channel_feat = feat_np[j]
            
            # 归一化特征图用于显示
            channel_feat_norm = (channel_feat - channel_feat.min()) / (channel_feat.max() - channel_feat.min() + 1e-8)
            
            axes[row, col].imshow(channel_feat_norm, cmap='viridis')
            axes[row, col].set_title(f'Ch {j+1}', fontsize=10)
            axes[row, col].axis('off')
        
        # 隐藏该行多余的列
        for j in range(num_display + 1, max_channels + 1):
            axes[row, j].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = 'assets/figures/resnet_feature_maps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"特征图可视化已保存至: {save_path}")


def visualize_activation_statistics(activations: Dict[str, np.ndarray],
                                  title: str = "激活值统计",
                                  save_path: Optional[str] = None) -> None:
    """
    可视化各层激活值的统计信息
    
    Args:
        activations: 字典 {layer_name: activations}
        title: 图表标题
        save_path: 保存路径
    """
    config = PlotConfig()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    layer_names = list(activations.keys())
    
    # 1. 激活值分布直方图
    ax1 = axes[0, 0]
    for layer_name, acts in activations.items():
        # 展平激活值
        acts_flat = acts.flatten()
        # 采样避免数据过多
        if len(acts_flat) > 10000:
            acts_flat = np.random.choice(acts_flat, 10000, replace=False)
        
        ax1.hist(acts_flat, bins=50, alpha=0.6, label=layer_name, density=True)
    
    ax1.set_xlabel('激活值')
    ax1.set_ylabel('密度')
    ax1.set_title('各层激活值分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 均值和方差
    ax2 = axes[0, 1]
    means = [np.mean(activations[name]) for name in layer_names]
    stds = [np.std(activations[name]) for name in layer_names]
    
    x = np.arange(len(layer_names))
    width = 0.35
    
    ax2.bar(x - width/2, means, width, label='均值', alpha=0.7)
    ax2.bar(x + width/2, stds, width, label='标准差', alpha=0.7)
    ax2.set_xlabel('网络层')
    ax2.set_ylabel('值')
    ax2.set_title('激活值统计')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 死神经元比例 (接近零的激活值比例)
    ax3 = axes[1, 0]
    dead_ratios = []
    for name, acts in activations.items():
        dead_ratio = np.mean(np.abs(acts) < 1e-6) * 100
        dead_ratios.append(dead_ratio)
    
    bars = ax3.bar(layer_names, dead_ratios, alpha=0.7, color='red')
    ax3.set_xlabel('网络层')
    ax3.set_ylabel('死神经元比例 (%)')
    ax3.set_title('死神经元统计')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for bar, ratio in zip(bars, dead_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ratio:.1f}%', ha='center', va='bottom')
    
    # 4. 激活值范围
    ax4 = axes[1, 1]
    mins = [np.min(activations[name]) for name in layer_names]
    maxs = [np.max(activations[name]) for name in layer_names]
    
    ax4.errorbar(layer_names, [(m + M) / 2 for m, M in zip(mins, maxs)], 
                yerr=[[(M - m) / 2 for m, M in zip(mins, maxs)]], 
                fmt='o', capsize=5, capthick=2)
    ax4.set_xlabel('网络层')
    ax4.set_ylabel('激活值')
    ax4.set_title('激活值范围')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = 'assets/figures/activation_statistics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"激活值统计图已保存至: {save_path}")


def visualize_all_conv_layers(conv_weights_dict: Dict[str, np.ndarray],
                           title: str = "LeNet-5 多层卷积核可视化",
                           save_path: Optional[str] = None,
                           max_kernels_per_layer: int = 8) -> None:
    """
    可视化多个卷积层的权重
    
    Args:
        conv_weights_dict: 卷积权重字典 {layer_name: weights}
        title: 图表标题
        save_path: 保存路径
        max_kernels_per_layer: 每层最多显示的卷积核数量
    """
    config = PlotConfig()
    
    num_layers = len(conv_weights_dict)
    fig, axes = plt.subplots(num_layers, max_kernels_per_layer, 
                           figsize=(max_kernels_per_layer * 3, num_layers * 2.5))
    
    if num_layers == 1:
        axes = [axes]  # 处理单层情况
    
    for row, (layer_name, weights) in enumerate(conv_weights_dict.items()):
        out_channels, in_channels, kh, kw = weights.shape
        num_display = min(max_kernels_per_layer, out_channels)
        
        if num_layers == 1:
            # 单层情况
            for col in range(num_display):
                ax = axes[col] if max_kernels_per_layer > 1 else axes
                
                if in_channels == 1:
                    kernel = weights[col, 0]
                else:
                    kernel = np.mean(weights[col], axis=0)
                
                im = ax.imshow(kernel, cmap='viridis', interpolation='nearest')
                ax.set_title(f'{layer_name}-K{col+1}', fontsize=10)
                ax.axis('off')
                
                if len(kernel.shape) == 2:
                    plt.colorbar(im, ax=ax, shrink=0.6)
            
            # 隐藏多余的子图
            for col in range(num_display, max_kernels_per_layer):
                axes[col].axis('off')
        else:
            # 多层情况
            for col in range(num_display):
                ax = axes[row, col]
                
                if in_channels == 1:
                    kernel = weights[col, 0]
                else:
                    kernel = np.mean(weights[col], axis=0)
                
                im = ax.imshow(kernel, cmap='viridis', interpolation='nearest')
                ax.set_title(f'{layer_name}-K{col+1}', fontsize=10)
                ax.axis('off')
                
                if len(kernel.shape) == 2:
                    plt.colorbar(im, ax=ax, shrink=0.6)
            
            # 隐藏该行多余的子图
            for col in range(num_display, max_kernels_per_layer):
                axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = 'assets/figures/lenet5_all_conv_kernels.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"多层卷积核可视化已保存至: {save_path}")


def create_kernel_gif(weights_history: List[np.ndarray],
                      layer_name: str = "Conv1",
                      save_path: Optional[str] = None) -> None:
    """
    创建卷积核演化的GIF动画 (需要imageio库)
    
    Args:
        weights_history: 训练过程中的权重历史
        layer_name: 层名称
        save_path: 保存路径
    """
    try:
        import imageio
        from PIL import Image
    except ImportError:
        print("需要安装 imageio 和 Pillow 来创建GIF: pip install imageio pillow")
        return
    
    config = PlotConfig()
    
    if save_path is None:
        save_path = f'../assets/figures/{layer_name.lower()}_evolution.gif'
    
    frames = []
    
    for epoch, weights in enumerate(weights_history):
        # 创建临时图像
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        axes = axes.flatten()
        
        out_channels = weights.shape[0]
        num_display = min(16, out_channels)
        
        for i in range(num_display):
            ax = axes[i]
            kernel = weights[i, 0]  # 假设单通道输入
            
            im = ax.imshow(kernel, cmap='viridis', interpolation='nearest')
            ax.set_title(f'K{i+1}', fontsize=8)
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_display, 16):
            axes[i].axis('off')
        
        plt.suptitle(f'{layer_name} 卷积核演化 - Epoch {epoch+1}', fontsize=14)
        plt.tight_layout()
        
        # 保存到内存
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        frames.append(Image.fromarray(image))
        
        plt.close()
    
    # 保存为GIF
    frames[0].save(save_path, save_all=True, append_images=frames[1:], 
                  duration=200, loop=0)
    
    print(f"卷积核演化GIF已保存至: {save_path}")


if __name__ == "__main__":
    # 测试探针功能
    print("测试探针功能...")
    
    # 测试卷积核可视化
    weights = np.random.randn(6, 1, 5, 5)  # 6个5x5卷积核
    visualize_conv_kernels(weights, "TestConv1")
    
    # 测试激活值统计
    activations = {
        'Conv1': np.random.randn(32, 24, 24) * 0.5,
        'Conv2': np.random.randn(64, 12, 12) * 0.3,
        'FC1': np.random.randn(128) * 0.1
    }
    visualize_activation_statistics(activations)
    
    print("探针功能测试完成！")
