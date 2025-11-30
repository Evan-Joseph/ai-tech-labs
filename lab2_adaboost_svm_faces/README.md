# 实验二：AdaBoost 与 SVM 人脸识别（复现实验指南）

本目录包含复现实验指标所需的最小代码与数据生成脚本。

## 环境准备

推荐使用 Conda（也可使用 pip 虚拟环境）。

```zsh
conda env create -f environment.yml
conda activate faces-lab
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
python experiments/pca_svm_analysis.py
```

说明：
- 将运行 PCA 降维与 SVM 分类器的核心实验。
- 包含不同核函数（Linear, RBF, Poly）的性能对比。
- 自动进行交叉验证与参数网格搜索。

产出：
- 图像在 `assets/figures/`
- 表格与数值在 `assets/tables/`

## 直接运行单个实验（可选）

```zsh
# 基础 PCA+SVM 在 ORL 数据集上的演示
python experiments/baseline_orl_pca_svm.py

# 挑战集测试（遮挡、光照变化等）
python experiments/demo_challenge_set.py

# 实时人脸检测与识别演示（需摄像头）
python experiments/demo_face_detection_recognition.py

# 训练识别器模型
python experiments/train_pca_svm_recognizer.py
```

## 复现提示

- 实验数据主要基于 ORL 人脸数据库，代码会自动处理数据加载。
- 挑战集测试需要预先准备相应的测试图像。
- 实时演示功能依赖于 OpenCV 和本地摄像头权限。
- 结果带有轻微随机性，已在代码中固定随机种子；如需统计稳健性，可多次运行取均值±标准差。
