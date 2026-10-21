# 🧠 AI Tech Labs (人工智能技术算法与模型复现基准)

<p align="center">
  <b>Reproducible Implementations of Classical Machine Learning & Deep Learning Paradigms</b><br>
  涵盖线性分类器、集成学习、经典卷积神经网络、深度残差网络 (Grad-CAM) 与时间序列多模型对比的系统化实验套件
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Git_LFS-Enabled-orange?logo=gitlfs&logoColor=white" alt="Git LFS">
  <img src="https://img.shields.io/badge/Reproducibility-100%25-brightgreen" alt="100% Reproducible">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

---

## 🖼️ 经典实验可视化画廊 (Visual Benchmarks & Output Gallery)

| Lab 1: 线性可分判别面迭代收敛 | Lab 2: 特征脸 (Eigenfaces) 空间正交基 |
| :---: | :---: |
| <img src="lab1_perceptron/assets/figures/pillar1_linear_separable_boundary.png" width="380" alt="PLA Decision Boundary" /> | <img src="lab2_adaboost_svm_faces/assets/figures/baseline_top_eigenfaces.png" width="380" alt="Top Eigenfaces Decomposition" /> |
| **感知器学习算法**：超平面动态逼近、间隔更新与两类样本分类边界演化。 | **主成分分析与正交分解**：高维人脸空间特征解耦与前 10 阶 Eigenfaces 重构。 |

| Lab 3: 深度卷积神经网络特征图演化 | Lab 4: Grad-CAM 空间注意力热力图 |
| :---: | :---: |
| <img src="lab3_lenet5_cnn/assets/figures/resnet_feature_maps.png" width="380" alt="CNN Feature Maps" /> | <img src="lab4_resnet_imagenet/assets/figures/cam_0.png" width="380" alt="Grad-CAM Saliency Map" /> |
| **层级语义表征提取**：从低层边缘纹理到高层类别语义的激活强度对比。 | **深度决策可解释性**：反向传播梯度加权类激活映射，精确定位图像核心判别区域。 |

<details>
<summary>📊 <b>点击展开查看：更多分类混淆矩阵与投影对比 (Click to view confusion matrices & projections)</b></summary>
<br>
<p align="center">
  <img src="lab1_perceptron/assets/figures/lda_faces_2d_projection.png" width="360" alt="LDA 2D Projection" style="margin: 6px;" />
  <img src="lab4_resnet_imagenet/assets/figures/confusion_matrix.png" width="360" alt="ResNet Confusion Matrix" style="margin: 6px;" />
</p>
</details>

---

## 📖 项目简介 (Overview)

**AI Tech Labs** 是一套面向高校与研究者的经典人工智能与深度学习算法复现基准库。

本项目摒弃了仅提供代码片段或论文空谈的模式，为 5 项代表性实验提供了完全**自包含、端到端可复现**的源码、微型评测集、环境依赖与可视化图表生成脚本。代码严格遵循模块化工程规范，是理解机器学习底层机理与深度模型演化的理想参考。

---

## 🔬 实验体系总览 (Laboratory Suite)

| 实验编号 | 课题名称 | 算法模型与理论 | 实验数据集 | 关键可视化与产出指标 |
|:---:|---|---|---|---|
| **Lab 1** | **感知器与线性分类器** | 感知器学习算法 (PLA)、多分类扩展、Fisher 线性判别分析 (LDA) | 人工合成 2D/3D 高斯聚类数据 | 决策分界面动态迭代收敛曲线、LDA 投影轴对比图 |
| **Lab 2** | **人脸识别与检测** | AdaBoost (弱分类器集成级联)、支持向量机 (Linear/RBF SVM) | LFW / ORL 人脸识别基准库 | ROC 曲线、混淆矩阵、滑动窗口检测热区 |
| **Lab 3** | **经典卷积神经网络** | 原始 LeNet-5 (Conv2D, MaxPool, AvgPool, 交叉熵) | MNIST 手写数字数据集 | 特征图 (Feature Maps) 激活层级演进图、Loss 曲线 |
| **Lab 4** | **深度残差网络与可解释性** | ResNet-18 / ResNet-50 残差跳跃连接、Grad-CAM 梯度加权类激活映射 | ImageNet / Tiny-ImageNet 子集 | Grad-CAM 空间注意力热力图、Top-1 / Top-5 准确率 |
| **Lab 5** | **序列建模与时间序列预测** | ARIMA、Facebook Prophet、LSTM、GRU、Temporal Transformer | 典型经济指标与电力负荷时间序列 | 多步预测轨迹对比、MAE / RMSE / MAPE 综合误差雷达图 |

---

## 📦 目录结构 (Directory Structure)

```text
ai-tech-labs/
├── lab1_perceptron/              # 线性可分与不可分数据集下的感知机实现
│   ├── experiments/              # 训练与分项测试脚本
│   └── assets/figures/           # 生成的分类面与投影可视化图表
├── lab2_adaboost_svm_faces/      # 特征抽取与人脸分类
│   ├── models/                   # 预训练权重 (Git LFS 托管)
│   └── assets/figures/
├── lab3_lenet5_cnn/              # PyTorch LeNet-5 骨干网络与自研梯度验证
├── lab4_resnet_imagenet/         # ResNet 迁移学习与 Grad-CAM 可解释性分析
├── lab5_lstm_rnn/                # 时间序列多模型基准对比评测套件
└── requirements.txt              # 统一 Python 依赖清单
```

---

## 🚀 快速开始与环境复现 (Quickstart)

### 1. 克隆代码与拉取模型文件 (Git LFS)

本项目使用 **Git LFS** 托管部分预训练权重与特征文件。克隆前请确认已配置 Git LFS：

```bash
git lfs install
git clone https://github.com/Evan-Joseph/ai-tech-labs.git
cd ai-tech-labs
git lfs pull
```

### 2. 创建并激活虚拟环境

```bash
# 推荐使用 Conda
conda create -n ai-labs python=3.11 -y
conda activate ai-labs

# 安装依赖
pip install -r requirements.txt
```

### 3. 一键运行任一实验

进入指定实验子目录，直接运行实验启动入口（以 Lab 4 Grad-CAM 为例）：

```bash
cd lab4_resnet_imagenet
python experiments/run_gradcam.py
```

实验生成的收敛图、热力图将自动保存至对应实验目录下的 `assets/figures/` 文件夹中。

---

## 📄 开源许可证 (License)

本项目采用 [MIT License](LICENSE) 开源，代码可自由用于教学、研究与二次开发。
