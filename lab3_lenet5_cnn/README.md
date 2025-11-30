# 实验三：卷积神经网络 LeNet-5 框架的设计实现及应用（复现实验指南）

本目录包含复现实验指标所需的最小代码与数据生成脚本。

## 环境准备

推荐使用 Conda（也可使用 pip 虚拟环境）。

```zsh
conda activate lenet-lab
```

或（可选）使用 pip：

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 一键复现实验

在本目录下执行：

```zsh
# 快速测试模式（5轮训练）
python experiments/run_comparison.py --quick-test

# 完整实验模式（20轮训练）
python experiments/run_comparison.py --epochs 20
```

说明：
- 将依次运行 LeNet-5 (NumPy), VGG (PyTorch), ResNet-18 (PyTorch) 三个模型。
- 自动进行训练、评估与对比分析。
- 生成损失曲线、混淆矩阵、特征图可视化等分析图表。

产出：
- 图像在 `assets/figures/`
- 表格与数值在 `assets/tables/`
- 训练日志在 `assets/logs/`

## 实验内容

### 1. LeNet-5 NumPy 底层复现
- **im2col 加速**：将卷积转换为矩阵乘法。
- **反向传播**：手动实现 Sigmoid 和卷积层的梯度计算。
- **架构适配**：针对 MNIST 28x28 输入调整网络结构。

### 2. 现代架构对比 (PyTorch)
- **VGG**：适配 MNIST 的轻量级 VGG 网络。
- **ResNet-18**：修改首层卷积与池化策略，保留空间特征。
- **性能对比**：收敛速度、准确率、训练效率的横向评估。

### 3. 网络内部探针
- **卷积核可视化**：展示 LeNet-5 学习到的边缘检测特征。
- **特征图演化**：追踪 ResNet 从浅层到深层的特征抽象过程。

## 复现提示

- LeNet-5 的 NumPy 实现训练较慢（约 5 分钟），这是正常的，旨在理解底层原理。
- PyTorch 模型（VGG/ResNet）建议使用 GPU (CUDA/MPS) 加速。
- 首次运行会自动下载 MNIST 数据集到 `data/` 目录。
- 结果带有轻微随机性，已在代码中固定随机种子。
