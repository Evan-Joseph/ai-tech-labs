#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.datasets import make_nearly_separable_with_noise, train_test_split_xy
from src.models.perceptron import Perceptron, PocketPerceptron, AveragedPerceptron, MarginPerceptron
from src.visualization.plotting import plot_data, plot_decision_boundary

ASSETS_FIG = Path(__file__).resolve().parents[1] / "assets" / "figures"
ASSETS_TAB = Path(__file__).resolve().parents[1] / "assets" / "tables"
ASSETS_FIG.mkdir(parents=True, exist_ok=True)
ASSETS_TAB.mkdir(parents=True, exist_ok=True)


def plot_margin_effects(X_train: np.ndarray, y_train: np.ndarray, mu: np.ndarray, sigma: np.ndarray, random_state: int = 42) -> None:
    """可视化不同余量值对带余量感知器决策边界的影响"""
    # 定义三个不同的余量值：小、中、大
    margin_values = [0.01, 1.0, 5.0]  # 小余量、中等余量、大余量
    margin_names = ["小余量 (b=0.01)", "中等余量 (b=1.0)", "大余量 (b=5.0)"]
    
    # 标准化训练数据
    X_train_std = (X_train - mu) / sigma
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    
    for i, (margin, name) in enumerate(zip(margin_values, margin_names)):
        # 训练带不同余量的感知器
        model = MarginPerceptron(
            learning_rate=1.0, 
            max_epochs=100, 
            shuffle=True, 
            random_state=random_state, 
            margin=margin
        )
        model.fit(X_train_std, y_train)
        
        # 将标准化空间中的参数转回原始坐标
        w_plot = model.w_ / sigma
        b_plot = float(model.b_ - np.dot(w_plot, mu))
        
        # 绘制数据点和决策边界
        plot_data(axes[i], X_train, y_train, title=name, legend=False)
        plot_decision_boundary(
            axes[i],
            X_train,
            decision_fn=model.decision_function,
            title=name,
            linear_params=(w_plot, b_plot),
            surface_params=(w_plot, b_plot),
        )
        
        # 添加余量信息
        axes[i].text(0.05, 0.95, f"余量: {margin}", transform=axes[i].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=10)
    
    fig.suptitle("不同余量(b)对带余量感知器决策边界的影响", y=1.05, fontsize=14)
    fig.tight_layout()
    fig.savefig(ASSETS_FIG / "pillar2_margin_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print(f"已生成余量影响分析图表: {ASSETS_FIG / 'pillar2_margin_effects.png'}")


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

    # 新增：不同余量值对决策边界影响的可视化
    plot_margin_effects(X_train, y_train, mu, sigma, random_state)

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
