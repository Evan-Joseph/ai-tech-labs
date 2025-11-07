#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.orl_faces import load_orl_faces
from src.visualization.plotting import plot_accuracy_curve

ASSETS_FIG = ROOT / "assets" / "figures"
ASSETS_TAB = ROOT / "assets" / "tables"
ASSETS_FIG.mkdir(parents=True, exist_ok=True)
ASSETS_TAB.mkdir(parents=True, exist_ok=True)


def _train_test_split(random_state: int = 0):
    faces = load_orl_faces(shuffle=True, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        faces.data,
        faces.target,
        test_size=0.30,
        stratify=faces.target,
        random_state=random_state,
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return {
        "faces": faces,
        "X_train": X_train_std,
        "X_test": X_test_std,
        "y_train": y_train,
        "y_test": y_test,
    }


def run_pca_dimension_study(random_state: int = 0) -> Dict[str, List[float]]:
    context = _train_test_split(random_state)
    components_grid = [10, 20, 40, 60, 80, 100, 120, 140, 160]
    accuracies: List[float] = []
    variance_retained: List[float] = []

    for n_components in components_grid:
        pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
        X_train_pca = pca.fit_transform(context["X_train"])
        X_test_pca = pca.transform(context["X_test"])
        svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=random_state)
        svm.fit(X_train_pca, context["y_train"])
        y_pred = svm.predict(X_test_pca)
        acc = accuracy_score(context["y_test"], y_pred)
        accuracies.append(acc * 100.0)
        variance_retained.append(float(pca.explained_variance_ratio_.sum()))

    df = pd.DataFrame(
        {
            "n_components": components_grid,
            "accuracy_percent": accuracies,
            "variance_retained": variance_retained,
        }
    )
    df.to_csv(ASSETS_TAB / "pca_dimension_accuracy.csv", index=False)
    df.to_json(ASSETS_TAB / "pca_dimension_accuracy.json", orient="records", indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    plot_accuracy_curve(
        ax,
        x=components_grid,
        y=accuracies,
        xlabel="保留主成分数量",
        ylabel="识别准确率（%）",
        title="保留主成分数量与识别准确率",
        marker="o",
    )
    ax.set_ylim(80, 100)
    fig.tight_layout()
    fig.savefig(ASSETS_FIG / "pca_components_vs_accuracy.png", dpi=200)
    plt.close(fig)

    return {
        "components": components_grid,
        "accuracy_percent": accuracies,
        "variance_retained": variance_retained,
    }


def run_kernel_comparison(random_state: int = 0) -> pd.DataFrame:
    context = _train_test_split(random_state)
    kernels = [
        {"name": "线性核", "kernel": "linear", "params": {"C": 10.0}},
        {"name": "多项式核", "kernel": "poly", "params": {"degree": 3, "C": 10.0, "gamma": "scale", "coef0": 1.0}},
        {"name": "RBF核", "kernel": "rbf", "params": {"C": 10.0, "gamma": "scale"}},
    ]

    records = []
    for cfg in kernels:
        pca = PCA(n_components=120, whiten=True, random_state=random_state)
        X_train_pca = pca.fit_transform(context["X_train"])
        X_test_pca = pca.transform(context["X_test"])
        svm = SVC(kernel=cfg["kernel"], random_state=random_state, **cfg["params"])
        svm.fit(X_train_pca, context["y_train"])
        y_pred = svm.predict(X_test_pca)
        acc = accuracy_score(context["y_test"], y_pred)
        records.append({
            "kernel_name": cfg["name"],
            "test_accuracy_percent": acc * 100.0,
        })

    df = pd.DataFrame(records)
    df_sorted = df.sort_values(by="test_accuracy_percent", ascending=False)
    df_sorted.to_csv(ASSETS_TAB / "svm_kernel_comparison.csv", index=False)
    df_sorted.to_json(ASSETS_TAB / "svm_kernel_comparison.json", orient="records", indent=2, force_ascii=False)

    df_tex = df_sorted.rename(columns={
        "kernel_name": "核函数",
        "test_accuracy_percent": "测试准确率（%）",
    })
    tex_table = df_tex.to_latex(
        index=False,
        float_format=lambda x: f"{x:.2f}",
        escape=True,
        caption="不同核函数在ORL数据集上的测试准确率对比",
        label="tab:svm-kernel-comparison",
    )
    tex_table = tex_table.replace("\\begin{table}", "\\begin{table}\n\\centering")
    (ASSETS_TAB / "svm_kernel_comparison.tex").write_text(tex_table, encoding="utf-8")

    return df_sorted


def run_cross_validation(random_state: int = 0) -> Dict[str, float]:
    faces = load_orl_faces(shuffle=True, random_state=random_state)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=120, whiten=True, random_state=random_state)),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=random_state)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, faces.data, faces.target, cv=cv, scoring="accuracy", n_jobs=None)
    scores_percent = scores * 100.0
    mean_acc = float(scores_percent.mean())
    std_acc = float(scores_percent.std(ddof=1))

    summary = {
        "fold_accuracies_percent": scores_percent.tolist(),
        "mean_accuracy_percent": mean_acc,
        "std_accuracy_percent": std_acc,
    }
    (ASSETS_TAB / "cross_validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    summary_df = pd.DataFrame(
        {
            "统计量": ["平均准确率", "标准差"],
            "数值（%）": [mean_acc, std_acc],
        }
    )
    summary_df.to_csv(ASSETS_TAB / "cross_validation_summary.csv", index=False)

    tex_cv = summary_df.to_latex(
        index=False,
        float_format=lambda x: f"{x:.2f}",
        escape=True,
        caption="5 折交叉验证的平均准确率与标准差",
        label="tab:cross-validation",
    )
    tex_cv = tex_cv.replace("\\begin{table}", "\\begin{table}\n\\centering")
    (ASSETS_TAB / "cross_validation_summary.tex").write_text(tex_cv, encoding="utf-8")

    return summary


def main(random_state: int = 0) -> None:
    pca_results = run_pca_dimension_study(random_state)
    kernels_df = run_kernel_comparison(random_state)
    cv_summary = run_cross_validation(random_state)

    print("PCA components tested:", pca_results["components"])
    print("Kernel comparison (top row):", kernels_df.iloc[0].to_dict())
    print("Cross-validation mean/std (%):", cv_summary["mean_accuracy_percent"], cv_summary["std_accuracy_percent"])


if __name__ == "__main__":
    main()
