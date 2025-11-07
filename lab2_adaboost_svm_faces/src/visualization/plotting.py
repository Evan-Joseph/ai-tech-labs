from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import seaborn as sns

# Consistent font stack: Times New Roman for Latin, Songti SC for Chinese
rcParams["font.family"] = "serif"
rcParams["font.serif"] = [
    "Songti SC",
    "STSong",
    "SimSun",
    "Times New Roman",
    "Times",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "DejaVu Serif",
]
rcParams["mathtext.fontset"] = "stix"
rcParams["axes.unicode_minus"] = False
rcParams["font.sans-serif"] = [
    "Songti SC",
    "PingFang SC",
    "STSong",
    "SimSun",
    "Times New Roman",
]

sns.set_theme(context="notebook", style="whitegrid", palette="deep", font="Songti SC")


def plot_confusion_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str = "",
    annotate: bool = True,
    label_step: int = 5,
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    normalized = matrix / row_sums

    heatmap = sns.heatmap(
        normalized,
        ax=ax,
        cmap="viridis",
        cbar=True,
        annot=False,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.85, "ticks": np.linspace(0.0, 1.0, 6)},
    )
    colorbar = heatmap.collections[0].colorbar
    if colorbar is not None:
        colorbar.ax.set_ylabel("Recall", rotation=-90, va="bottom")

    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)

    tick_positions = np.arange(len(labels))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    xtick_labels = [label if idx % label_step == 0 else "" for idx, label in enumerate(labels)]
    ytick_labels = [label if idx % label_step == 0 else "" for idx, label in enumerate(labels)]
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_yticklabels(ytick_labels, rotation=0)


def plot_eigenfaces(
    ax_array: Iterable[plt.Axes],
    faces: np.ndarray,
    image_shape: tuple[int, int],
    *,
    titles: Sequence[str] | None = None,
    cmap: str = "coolwarm",
) -> None:
    faces = np.asarray(faces)
    for idx, (ax, face) in enumerate(zip(ax_array, faces)):
        ax.imshow(face.reshape(image_shape), cmap=cmap)
        ax.axis("off")
        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)


def plot_accuracy_curve(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    marker: str = "o",
) -> None:
    ax.plot(x, y, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
