#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.datasets import make_linearly_separable
from src.models.perceptron import Perceptron
from src.visualization.plotting import plot_data, plot_decision_boundary, plot_errors_curve

ASSETS_FIG = Path(__file__).resolve().parents[1] / "assets" / "figures"
ASSETS_FIG.mkdir(parents=True, exist_ok=True)


def main(random_state: int = 42) -> None:
    X, y = make_linearly_separable(n_samples=200, noise=0.0, random_state=random_state)

    clf = Perceptron(learning_rate=1.0, max_epochs=200, shuffle=True, random_state=random_state)
    clf.fit(X, y)

    # Plot training errors over epochs
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_errors_curve(ax, clf.history_.errors_per_epoch, title="线性可分：错分点数量随迭代变化（应收敛至 0）")
    fig.tight_layout()
    fig.savefig(ASSETS_FIG / "pillar1_linear_separable_errors.png", dpi=200)
    plt.close(fig)

    # Plot data and decision boundary
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plot_data(ax, X, y, title="线性可分数据")
    plot_decision_boundary(
        ax,
        X,
        decision_fn=clf.decision_function,
        title="线性可分：最终决策边界",
        linear_params=(clf.w_, clf.b_),
        surface_params=(clf.w_, clf.b_),
    )
    fig.tight_layout()
    fig.savefig(ASSETS_FIG / "pillar1_linear_separable_boundary.png", dpi=200)
    plt.close(fig)

    # Print convergence info
    print(f"Epochs ran: {clf.history_.epochs_ran}")
    print(f"Final training errors: {clf.history_.errors_per_epoch[-1] if clf.history_.errors_per_epoch else None}")


if __name__ == "__main__":
    main()
