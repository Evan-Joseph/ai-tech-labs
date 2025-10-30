# 实验一：感知器线性分类器（复现实验指南）

本目录包含复现实验指标所需的最小代码与数据生成脚本。

## 环境准备

推荐使用 Conda（也可使用 pip 虚拟环境）。

```zsh
conda env create -f environment.yml
conda activate perceptron-lab
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
- 将依次运行二分类三组实验（线性可分、XOR、变体比较与 SVM 对比）。
- 然后运行多类高斯数据的最大值判决感知器；
- 最后运行 LDA+感知器人脸识别（首次会下载 Olivetti 数据，需联网）。

产出：
- 图像在 `assets/figures/`
- 表格与数值在 `assets/tables/`

## 直接运行单个实验（可选）

```zsh
# 线性可分/不可分复现
python experiments/pillar1_linear_separable.py
python experiments/pillar1_xor_nonseparable.py

# 含噪近似可分：感知器变体对比
python experiments/pillar2_comparisons.py

# 平均感知器 vs LinearSVC 对比
python experiments/pillar3_svm_compare.py

# 多类：最大值判决（Kesler 构造）
python experiments/multiclass_perceptron_demo.py

# 拓展：LDA 降维 + 感知器（首次需联网下载数据）
python experiments/lda_faces_perceptron.py
```

## 复现提示

- 若在非 Conda 环境，确保 Python 版本与 `environment.yml` 一致（推荐 3.11）。
- 人脸数据实验首次会下载数据集到本机缓存路径，时间稍长属正常现象。
- 结果带有轻微随机性，已在代码中固定随机种子；如需统计稳健性，可多次运行取均值±标准差。
