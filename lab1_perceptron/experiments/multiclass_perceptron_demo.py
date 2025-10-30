#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
# ensure project root on sys.path for `import src.*`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import make_gaussian_multiclass, train_test_split_xy
from src.models.multiclass_perceptron import MultiClassPerceptronMax
from src.visualization.plotting import plot_multiclass_data, plot_multiclass_decision_regions

ASSETS_FIG = ROOT / "assets" / "figures"
ASSETS_TAB = ROOT / "assets" / "tables"
ASSETS_FIG.mkdir(parents=True, exist_ok=True)
ASSETS_TAB.mkdir(parents=True, exist_ok=True)


def main(random_state: int = 7) -> None:
    # 生成 N>=6 类的 2D 高斯数据
    X, y = make_gaussian_multiclass(n_classes=6, n_samples_per_class=90, n_features=2, radius=4.0, scale=0.8, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y, test_size=0.30, random_state=random_state)

    clf = MultiClassPerceptronMax(learning_rate=1.0, max_epochs=200, shuffle=True, random_state=random_state, margin=0.0)
    clf.fit(X_train, y_train)

    # 训练集多类决策区域可视化
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    plot_multiclass_decision_regions(ax, X_train, predict_fn=clf.predict, title="多类感知器（最大值判决）决策区域（训练集）")
    plot_multiclass_data(ax, X_train, y_train, title="多类感知器（最大值判决）决策区域（训练集）", legend=True)
    fig.tight_layout()
    fig.savefig(ASSETS_FIG / "multiclass_perceptron_regions.png", dpi=200)
    plt.close(fig)

    # 测试集准确率与混淆矩阵
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 保存结果
    np.savetxt(ASSETS_TAB / "multiclass_confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")
    with open(ASSETS_TAB / "multiclass_summary.json", "w", encoding="utf-8") as f:
        json.dump({"test_accuracy": float(acc)}, f, indent=2, ensure_ascii=False)

    # 混淆矩阵可视
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(f"多类感知器 混淆矩阵 (测试准确率={acc:.3f})")
        fig.tight_layout()
        fig.savefig(ASSETS_FIG / "multiclass_confusion_matrix.png", dpi=200)
        plt.close(fig)
    except Exception:
        pass

    print(f"Test accuracy (multiclass perceptron): {acc:.4f}")


if __name__ == "__main__":
    main()
