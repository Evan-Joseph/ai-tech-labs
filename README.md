# 人工智能技术课程实验

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Last commit](https://img.shields.io/github/last-commit/Evan-Joseph/ai-tech-labs?color=brightgreen)
![Issues](https://img.shields.io/github/issues/Evan-Joseph/ai-tech-labs)
![Stars](https://img.shields.io/github/stars/Evan-Joseph/ai-tech-labs?style=social)

本仓库是为了配合《人工智能技术》课程的实验报告而专门设置。仓库仅包含可复现实验的代码与产出（图/表），不包含报告（LaTeX/论文）文件。

## 实验总览（5 个实验，每个一个文件夹）

1. `lab1_perceptron/` 感知器线性分类器的设计实现（含二分类、多类与 LDA 拓展）
2. `lab2_adaboost_svm_faces/` 基于 Adaboost 及 SVM 的人脸识别程序的设计实现（占位，待补充）
3. `lab3_lenet5_cnn/` 卷积神经网络 LeNet-5 框架的设计实现及应用（占位，待补充）
4. `lab4_resnet_imagenet/` ResNet 神经网络实现 ImageNet 图像分类（占位，待补充）
5. `lab5_lstm_rnn/` 循环神经网络 LSTM 的实现及应用（占位，待补充）

> 每个实验具体内容见对应子目录 `README.md`。

## 统一复现方式

进入对应实验子目录，阅读 `README.md` 并按指引：
- 创建/激活环境（conda 或 pip）
- 运行一键脚本 `experiments/run_all.py` 或分项脚本
- 结果图表将保存到该实验的 `assets/` 目录
