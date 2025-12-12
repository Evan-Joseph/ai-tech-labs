# 实验五：基于 LSTM/RNN 的时间序列预测（复现实验指南）

本目录包含复现实验指标所需的最小代码与数据生成脚本。

## 环境准备

推荐使用 Conda（也可使用 pip 虚拟环境）。

```zsh
conda activate ai_lab_05_lstm
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
- 加载高密市国有土地发证记录数据（3,074 条）。
- 按月聚合构建 110 个月的时间序列。
- 依次训练 ARIMA, Prophet, LSTM, GRU, Transformer 五种模型。
- 自动进行训练、评估与对比分析。
- 生成时序趋势图、预测对比图、训练损失曲线等分析图表。

产出：
- 图像在 `assets/figures/`
- 表格与数值在 `assets/tables/`
- 实验摘要在 `assets/experiment_summary.txt`

## 实验内容

### 1. 数据工程 (Data Engineering)
- **月度聚合**：将原始发证记录按月统计数量。
- **ADF 平稳性检验**：验证序列平稳性（统计量 -5.68, p < 0.0001）。
- **滑动窗口构建**：使用 6 个月历史数据预测下 1 个月。

### 2. 多模型对比
- **基线模型**：ARIMA（自动参数搜索）、Prophet（年度季节性）
- **核心模型**：LSTM、GRU（门控循环单元）
- **探索模型**：Transformer（自注意力机制）

### 3. 核心结论
在高波动小样本政务数据场景下，**门控循环网络（LSTM/GRU）表现最优**：

| 模型 | MAE | RMSE | 特点 |
|------|-----|------|------|
| **GRU** | **14.45** | 19.94 | ✅ MAE 最优，结构简洁 |
| **LSTM** | 14.83 | **19.75** | ✅ RMSE 最优，门控结构 |
| Transformer | 15.52 | 20.01 | 表现接近，但易过拟合 |
| ARIMA | 15.72 | 20.31 | 统计模型，稳健 |
| Prophet | 20.52 | 26.32 | MAPE 最优 (87.77%) |

## 超参数优化要点

为避免"模式崩溃"（模型预测恒定值），采用以下优化策略：
- **减少层数**：从 2 层减少到 1 层，避免过度正则化
- **增大隐藏层**：从 32 增大到 64，提升表达能力
- **提高学习率**：从 0.001 提高到 0.005，加速收敛
- **缩短窗口**：从 12 个月缩短到 6 个月，增加训练样本

## 模型选择建议

| 应用场景 | 推荐模型 |
|----------|----------|
| 小样本高波动数据 | GRU 或 LSTM |
| 需要可解释性 | ARIMA 或 Prophet |
| 趋势预测 | Prophet（MAPE 最优） |
| 大规模数据 | Transformer（需充分正则化） |

## 复现提示

- 数据文件位于 `data/raw/国有土地发证记录.csv`（GB18030 编码）。
- 推荐使用 GPU (如 Apple MPS 或 NVIDIA CUDA) 加速训练。
- 深度学习模型配置了早停机制，训练通常在 30-50 个 Epoch 内完成。
- 如需调整超参数，请修改 `experiments/run_all.py` 中的 `CONFIG` 字典。

## 目录结构

```
lab5_lstm_rnn/
├── src/
│   ├── data/           # 数据加载与预处理
│   ├── models/         # ARIMA, Prophet, LSTM, GRU, Transformer
│   ├── visualization/  # 可视化模块
│   └── utils/          # 评估指标
├── experiments/
│   └── run_all.py      # 一键执行脚本
├── assets/
│   ├── figures/        # 可视化图表
│   └── tables/         # LaTeX 表格
├── docs/report/
│   └── main.tex        # LaTeX 实验报告
└── data/raw/           # 原始数据
```
