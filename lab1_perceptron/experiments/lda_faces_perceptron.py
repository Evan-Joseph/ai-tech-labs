#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import json

ROOT = Path(__file__).resolve().parents[1]
# ensure project root on sys.path for `import src.*`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.models.multiclass_perceptron import MultiClassPerceptronMax
from src.visualization.plotting import plot_multiclass_data

ASSETS_FIG = ROOT / "assets" / "figures"
ASSETS_TAB = ROOT / "assets" / "tables"
ASSETS_FIG.mkdir(parents=True, exist_ok=True)
ASSETS_TAB.mkdir(parents=True, exist_ok=True)


def main(random_state: int = 0) -> None:
    # 加载 ORL/Olivetti 人脸数据（需要网络下载一次）
    data = fetch_olivetti_faces(shuffle=True, random_state=random_state)
    X = data.data  # (400, 4096)
    y = data.target  # 40 类，每类 10 张

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    # 标准化到零均值单位方差
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Baseline: 直接在像素空间训练多类感知器
    base_clf = MultiClassPerceptronMax(learning_rate=0.1, max_epochs=50, shuffle=True, random_state=random_state, margin=0.0)
    base_clf.fit(X_train_std, y_train)
    y_pred_base = base_clf.predict(X_test_std)
    acc_base = accuracy_score(y_test, y_pred_base)

    # LDA 降维到 K-1 维（K=40 -> 39 维），再训练多类感知器
    lda = LDA(solver="svd")
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    X_test_lda = lda.transform(X_test_std)

    lda_clf = MultiClassPerceptronMax(learning_rate=0.1, max_epochs=50, shuffle=True, random_state=random_state, margin=0.0)
    lda_clf.fit(X_train_lda, y_train)
    y_pred_lda = lda_clf.predict(X_test_lda)
    acc_lda = accuracy_score(y_test, y_pred_lda)

    # 记录结果
    results = {
        "baseline_pixel_perceptron": float(acc_base),
        "lda_plus_perceptron": float(acc_lda),
    }
    with open(ASSETS_TAB / "lda_faces_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 可视化：LDA 前两维散点
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        # 仅绘制训练集的前两维
        X2 = X_train_lda[:, :2]
        # 复用多类散点绘制
        plot_multiclass_data(ax, X2, y_train, title="LDA 前两维投影（训练集）", legend=False)
        fig.tight_layout()
        fig.savefig(ASSETS_FIG / "lda_faces_2d_projection.png", dpi=200)
        plt.close(fig)
    except Exception:
        pass

    # 混淆矩阵（基于 LDA+感知器）
    cm = confusion_matrix(y_test, y_pred_lda)
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        sns.heatmap(cm, cmap="Blues", cbar=True)
        ax.set_title(f"LDA+感知器 混淆矩阵 (测试准确率={acc_lda:.3f})")
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        fig.tight_layout()
        fig.savefig(ASSETS_FIG / "lda_faces_confusion_matrix.png", dpi=200)
        plt.close(fig)
    except Exception:
        pass

    print("Results:")
    print(results)


if __name__ == "__main__":
    main()
