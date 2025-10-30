#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from src.data.datasets import make_nearly_separable_with_noise, train_test_split_xy
from src.models.perceptron import AveragedPerceptron
from src.visualization.plotting import plot_data, plot_decision_boundary

ASSETS_FIG = Path(__file__).resolve().parents[1] / "assets" / "figures"
ASSETS_TAB = Path(__file__).resolve().parents[1] / "assets" / "tables"
ASSETS_FIG.mkdir(parents=True, exist_ok=True)
ASSETS_TAB.mkdir(parents=True, exist_ok=True)


def main(random_state: int = 42) -> None:
    # Use the SAME scenario as pillar 2 for strict comparability
    X, y = make_nearly_separable_with_noise(n_samples=400, noise=0.4, flip_y=0.10, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y, test_size=0.3, random_state=random_state)

    # Standardize for stable training on noisy data (match pillar 2)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    ap = AveragedPerceptron(learning_rate=1.0, max_epochs=15, shuffle=True, random_state=random_state)
    ap.fit(X_train_std, y_train)

    svm = LinearSVC(C=1.0, random_state=random_state, max_iter=5000)
    svm.fit(X_train_std, (y_train > 0).astype(int))  # LinearSVC expects labels as classes, map {-1,+1}->{0,1}

    # Plot side-by-side decision boundaries
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)

    # Averaged Perceptron
    plot_data(axes[0], X_train, y_train, title="平均感知器（训练集）", legend=False)
    plot_decision_boundary(
        axes[0],
        X_train,
        decision_fn=lambda X: ap.decision_function((X - mu) / sigma, use_average=True),
        title="平均感知器 决策边界",
        # transform params back to original x-space
        linear_params=(ap.w_avg_ / sigma, float(ap.b_avg_ - (ap.w_avg_ / sigma) @ mu)),
        surface_params=(ap.w_avg_ / sigma, float(ap.b_avg_ - (ap.w_avg_ / sigma) @ mu)),
    )

    # LinearSVC
    def svm_decision(X):
        # decision in standardized space
        return (X - mu) / sigma @ svm.coef_.ravel() + svm.intercept_.item()

    plot_data(axes[1], X_train, y_train, title="LinearSVC（训练集）", legend=False)
    plot_decision_boundary(
        axes[1],
        X_train,
        decision_fn=svm_decision,
        title="LinearSVC 决策边界",
        # map standardized-space hyperplane back to original coordinates
        linear_params=(svm.coef_.ravel() / sigma, float(svm.intercept_.ravel()[0] - (svm.coef_.ravel() / sigma) @ mu)),
        surface_params=(svm.coef_.ravel() / sigma, float(svm.intercept_.ravel()[0] - (svm.coef_.ravel() / sigma) @ mu)),
    )

    fig.suptitle("平均感知器 vs. LinearSVC 决策边界对比（最大间隔更鲁棒）", y=1.03)
    fig.tight_layout()
    fig.savefig(ASSETS_FIG / "pillar3_ap_vs_svm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Accuracy on test
    y_pred_ap = ap.predict(X_test_std, use_average=True)
    y_pred_svm = svm.predict(X_test_std)
    y_true_01 = (y_test > 0).astype(int)

    acc_ap = accuracy_score(y_true_01, (y_pred_ap > 0).astype(int))
    acc_svm = accuracy_score(y_true_01, y_pred_svm)

    with open(ASSETS_TAB / "pillar3_ap_vs_svm.txt", "w") as f:
        f.write(f"AveragedPerceptron test acc: {acc_ap:.4f}\n")
        f.write(f"LinearSVC test acc:        {acc_svm:.4f}\n")


if __name__ == "__main__":
    main()
