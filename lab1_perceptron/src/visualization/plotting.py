from __future__ import annotations

from typing import Optional, Tuple, List, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

# Set seaborn style and ensure Chinese fonts render in saved figures
sns.set(context="notebook", style="whitegrid", palette="deep")
# Font policy: Times New Roman (Latin) + 宋体 (Chinese)
rcParams["font.family"] = "serif"
rcParams["font.serif"] = [
    "Songti SC",     # 宋体 (macOS)
    "SimSun",        # 宋体 (Windows)
    "STSong",
    "Times New Roman",
    "Times",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "DejaVu Serif",
]
rcParams["mathtext.fontset"] = "stix"
rcParams["axes.unicode_minus"] = False


def plot_data(ax: plt.Axes, X: np.ndarray, y: np.ndarray, title: str = "", legend: bool = True) -> None:
    y = np.asarray(y)
    pos = y > 0
    neg = ~pos
    ax.scatter(X[pos, 0], X[pos, 1], c="#2ca02c", label="正类 (+1)", s=30, alpha=0.8)
    ax.scatter(X[neg, 0], X[neg, 1], c="#d62728", label="负类 (-1)", s=30, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    if legend:
        ax.legend(loc="best")


def _grid(X: np.ndarray, padding: float = 1.0, h: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_decision_boundary(
    ax: plt.Axes,
    X: np.ndarray,
    decision_fn: Callable[[np.ndarray], np.ndarray],
    title: str = "",
    levels: Optional[List[float]] = None,
    linear_params: Optional[Tuple[np.ndarray, float]] = None,
    surface_params: Optional[Tuple[np.ndarray, float]] = None,
) -> None:
    """Plot decision function background and draw boundary.

    If the zero level-set is outside the grid (no contour at 0), and
    linear_params=(w, b) is provided, draw the straight line w·x + b = 0.
    """
    xx, yy, grid = _grid(X)
    # Use explicit (w, b) for the decision surface if provided to ensure
    # the background and the drawn straight line are computed from the same
    # parameters (e.g., Pocket/Averaged Perceptron).
    if surface_params is not None:
        w_surf, b_surf = surface_params
        Z = (grid @ w_surf + b_surf).reshape(xx.shape)
    else:
        Z = decision_fn(grid).reshape(xx.shape)
    # contourf background with color center aligned to 0 for visual consistency
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    zmin, zmax = float(np.min(Z)), float(np.max(Z))
    # ensure vmin < 0 < vmax for TwoSlopeNorm; pad slightly if needed
    eps = 1e-9
    vmin = zmin if zmin < 0 else -eps
    vmax = zmax if zmax > 0 else eps
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    ax.contourf(xx, yy, Z, levels=50, cmap=cmap, alpha=0.30, norm=norm)

    # Try to draw the decision boundary at 0
    contour_levels = [0.0] if levels is None else levels
    cs = ax.contour(xx, yy, Z, levels=contour_levels, colors="k", linewidths=2)

    # Fallback: draw analytical straight line for linear models
    if (levels is None) and (len(cs.allsegs[0]) == 0) and (linear_params is not None):
        w, b = linear_params
        if w is not None:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            # vertical vs non-vertical
            if abs(w[1]) > 1e-12:
                xs = np.array([x_min, x_max])
                ys = (-b - w[0] * xs) / w[1]
                ax.plot(xs, ys, color="k", linewidth=2)
            elif abs(w[0]) > 1e-12:
                x0 = -b / w[0]
                ax.axvline(x0, color="k", linewidth=2)
            # else: degenerate, skip

    ax.set_title(title)


def plot_errors_curve(ax: plt.Axes, errors: List[int], title: str) -> None:
    ax.plot(np.arange(1, len(errors) + 1), errors, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassified points")
    ax.set_title(title)
    ax.grid(True)


def plot_multiclass_data(ax: plt.Axes, X: np.ndarray, y: np.ndarray, title: str = "", legend: bool = True) -> None:
    """Scatter plot for multi-class labels (0..K-1)."""
    y = np.asarray(y)
    classes = np.unique(y)
    palette = sns.color_palette(n_colors=len(classes))
    for cls, color in zip(classes, palette):
        mask = (y == cls)
        ax.scatter(X[mask, 0], X[mask, 1], s=25, alpha=0.85, label=f"类 {cls}", color=color)
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    if legend:
        ax.legend(loc="best", ncol=2)


def plot_multiclass_decision_regions(
    ax: plt.Axes,
    X: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    title: str = "",
    alpha_bg: float = 0.25,
) -> None:
    """Plot multiclass decision regions by coloring argmax predictions on a grid."""
    xx, yy, grid = _grid(X)
    y_hat = predict_fn(grid).ravel()
    classes = np.unique(y_hat)
    palette = sns.color_palette(n_colors=len(classes))
    # create color map for classes
    class_to_color = {c: palette[i] for i, c in enumerate(classes)}
    # colorize background
    Z_rgb = np.array([class_to_color[c] for c in y_hat]).reshape((*xx.shape, 3))
    ax.imshow(
        Z_rgb,
        origin="lower",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        alpha=alpha_bg,
    )
    ax.set_title(title)
