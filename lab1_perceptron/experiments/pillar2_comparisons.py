#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.data.datasets import make_nearly_separable_with_noise, train_test_split_xy
from src.models.perceptron import Perceptron, PocketPerceptron, AveragedPerceptron, MarginPerceptron
from src.visualization.plotting import plot_data, plot_decision_boundary

ASSETS_FIG = Path(__file__).resolve().parents[1] / "assets" / "figures"
ASSETS_TAB = Path(__file__).resolve().parents[1] / "assets" / "tables"
ASSETS_FIG.mkdir(parents=True, exist_ok=True)
ASSETS_TAB.mkdir(parents=True, exist_ok=True)


def eval_and_plot_models(random_state: int = 42) -> None:
    # Data: use more distinctive nearly-separable setting
    X, y = make_nearly_separable_with_noise(n_samples=400, noise=0.4, flip_y=0.10, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y, test_size=0.3, random_state=random_state)

    # 标准化：提高在含噪声数据上的稳定性（训练在标准化空间进行）
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    models = {
        "Perceptron": Perceptron(learning_rate=1.0, max_epochs=100, shuffle=True, random_state=random_state),
        "PocketPerceptron": PocketPerceptron(learning_rate=1.0, max_epochs=100, shuffle=True, random_state=random_state),
        "AveragedPerceptron": AveragedPerceptron(learning_rate=1.0, max_epochs=15, shuffle=True, random_state=random_state),
        "MarginPerceptron": MarginPerceptron(learning_rate=1.0, max_epochs=100, shuffle=True, random_state=random_state, margin=1.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    results = []
    for ax, (name, model) in zip(axes, models.items()):
        # 在标准化特征上训练
        model.fit(X_train_std, y_train)
        if isinstance(model, AveragedPerceptron):
            decision_fn = lambda X: model.decision_function(X, use_average=True)
        else:
            decision_fn = model.decision_function

        plot_data(ax, X_train, y_train, title=f"{name}（训练集）", legend=False)
        # 将标准化空间中的 (w,b) 转回原始坐标，用于绘图与背景
        if isinstance(model, AveragedPerceptron):
            w_use, b_use = model.w_avg_, model.b_avg_
        else:
            w_use, b_use = model.w_, model.b_
        w_plot = w_use / sigma
        b_plot = float(b_use - np.dot(w_plot, mu))
        plot_decision_boundary(
            ax,
            X_train,
            decision_fn=decision_fn,
            title=f"{name}",
            linear_params=(w_plot, b_plot),
            surface_params=(w_plot, b_plot),
        )

        # accuracy
        if isinstance(model, AveragedPerceptron):
            y_pred = model.predict(X_test_std, use_average=True)
        else:
            y_pred = model.predict(X_test_std)
        acc = accuracy_score(y_test, y_pred)
        results.append({"model": name, "test_accuracy": float(acc)})

    fig.suptitle("带少量噪声近似线性可分数据：四种算法决策边界对比", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS_FIG / "pillar2_boundaries_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save table (CSV + JSON)
    import pandas as pd
    df = pd.DataFrame(results).sort_values(by="test_accuracy", ascending=False)
    df.to_csv(ASSETS_TAB / "pillar2_accuracy.csv", index=False)

    # Rename columns to avoid LaTeX underscore issues and use Chinese headers
    df_latex = df.rename(columns={
        "model": "模型",
        "test_accuracy": "测试准确率",
    })
    # Save simple LaTeX table for inclusion (ensure escaping enabled)
    tex_table = df_latex.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        escape=True,
        caption="四种感知器变体在测试集上的准确率对比",
        label="tab:pillar2-acc",
    )
    # center the table and ensure booktabs three-line style is used by pandas default
    tex_table = tex_table.replace("\\begin{table}", "\\begin{table}\n\\centering")
    (ASSETS_TAB / "pillar2_accuracy.tex").write_text(tex_table)

    # JSON summary
    (ASSETS_TAB / "pillar2_accuracy.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    eval_and_plot_models()
