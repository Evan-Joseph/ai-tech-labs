# 实验四：经典卷积神经网络实现 ImageNet 图像分类（复现实验指南）

本目录包含复现实验指标所需的最小代码与数据生成脚本。

## 环境准备

推荐使用 Conda（也可使用 pip 虚拟环境）。

```zsh
conda activate ai_lab_04_resnet
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
python experiments/run_all.py
```

说明：
- 自动下载并解压 ImageNette (ImageNet 子集) 数据集。
- 依次运行 AlexNet, VGG16, ResNet50 三个模型的迁移学习微调。
- 自动进行训练、评估与对比分析。
- 生成模型效率对比图、混淆矩阵、Grad-CAM 热力图等分析图表。

产出：
- 图像在 `assets/figures/`
- 表格与数值在 `assets/tables/`
- 模型检查点在 `assets/checkpoints/`

## 实验内容

### 1. 迁移学习 (Transfer Learning)
- **预训练权重**：加载 ImageNet 上预训练的官方权重。
- **层级冻结**：冻结特征提取层，仅训练全连接分类层。
- **数据适配**：实现 ImageNet 标准预处理流水线 (Resize, CenterCrop, Normalize)。

### 2. 经典架构对比
- **AlexNet**：深度学习破冰之作，验证 ReLU 与 Dropout 的有效性。
- **VGG16**：深层网络代表，探究堆叠小卷积核对特征提取的影响。
- **ResNet50**：残差网络代表，验证残差连接如何解决深层网络退化问题，实现参数效率与性能的双重提升。

### 3. 可解释性分析 (Grad-CAM)
- **注意力可视化**：生成梯度加权类激活映射 (Grad-CAM) 热力图。
- **语义定位验证**：验证模型是否关注图像中的关键主体（如鱼、狗）而非背景噪声，对比不同架构的语义定位能力。

## 复现提示

- 数据集较大（约 150MB），首次运行需要下载，请保持网络通畅。
- 推荐使用 GPU (如 Apple MPS 或 NVIDIA CUDA) 加速训练，否则 VGG16 的推理可能较慢。
- 默认训练 5 个 Epoch，因迁移学习收敛极快，这已足以达到 99% 准确率。
