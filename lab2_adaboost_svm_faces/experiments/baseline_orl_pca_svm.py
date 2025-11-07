#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.orl_faces import load_orl_faces
from src.models.baseline import run_pca_svm_baseline
from src.visualization.plotting import plot_confusion_matrix, plot_eigenfaces

ASSETS_FIG = ROOT / "assets" / "figures"
ASSETS_TAB = ROOT / "assets" / "tables"
ASSETS_FIG.mkdir(parents=True, exist_ok=True)
ASSETS_TAB.mkdir(parents=True, exist_ok=True)


def main(random_state: int = 0) -> None:
    faces = load_orl_faces(shuffle=True, random_state=random_state)
    result = run_pca_svm_baseline(
        faces.data,
        faces.target,
        test_size=0.3,
        random_state=random_state,
        n_components=120,
        kernel="rbf",
        C=10.0,
        gamma="scale",
    )

    summary = {
        "accuracy": result.accuracy,
        "train_samples": result.train_shape[0],
        "test_samples": result.test_shape[0],
        "n_components": int(result.pca_components.shape[0]),
    }
    (ASSETS_TAB / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Confusion matrix figure
    fig_cm, ax_cm = plt.subplots(figsize=(7.0, 6.0))
    label_text = [f"ID {i:02d}" for i in np.unique(result.y_true)]
    plot_confusion_matrix(
        ax_cm,
        result.confusion,
        labels=label_text,
        title="",
        annotate=False,
        label_step=5,
    )
    fig_cm.tight_layout()
    fig_cm.savefig(ASSETS_FIG / "baseline_confusion_matrix.png", dpi=200)
    plt.close(fig_cm)

    # Eigenfaces figure (前 10 个特征脸)
    top_k = 10
    eigenfaces = result.pca_components[:top_k]
    cols = 5
    rows = int(np.ceil(top_k / cols))
    fig_faces, axes_faces = plt.subplots(rows, cols, figsize=(12, 5))
    axes = axes_faces.ravel()
    titles = [f"PC {i+1}" for i in range(top_k)]
    plot_eigenfaces(axes[:top_k], eigenfaces, faces.image_shape, titles=titles)
    for extra_ax in axes[top_k:]:
        extra_ax.axis("off")
    fig_faces.tight_layout()
    fig_faces.savefig(ASSETS_FIG / "baseline_top_eigenfaces.png", dpi=200)
    plt.close(fig_faces)

    print("Baseline accuracy:", result.accuracy)
    print("Summary saved to:", ASSETS_TAB / "baseline_summary.json")


if __name__ == "__main__":
    main()
