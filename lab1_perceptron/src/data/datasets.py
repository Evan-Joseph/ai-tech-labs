from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split


def make_linearly_separable(n_samples: int = 200, noise: float = 0.0, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 2D linearly separable dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples (will be split equally between two classes).
    noise : float
        Gaussian noise std added to features.
    random_state : int
        RNG seed.
    """
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2

    # Two Gaussian blobs separated by a line
    mean_pos = np.array([2.0, 2.0])
    mean_neg = np.array([-2.0, -2.0])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])

    X_pos = rng.multivariate_normal(mean_pos, cov, size=n_half)
    X_neg = rng.multivariate_normal(mean_neg, cov, size=n_samples - n_half)

    if noise > 0:
        X_pos += rng.normal(0, noise, size=X_pos.shape)
        X_neg += rng.normal(0, noise, size=X_neg.shape)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])
    return X, y


def make_xor(n_samples: int = 200, noise: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 2D XOR dataset with optional Gaussian noise.

    Quadrants (x1, x2):
      - (+,+) and (-,-) -> class +1
      - (+,-) and (-,+) -> class -1
    """
    rng = np.random.default_rng(random_state)
    n_quarter = n_samples // 4

    def blob(center):
        return center + rng.normal(0, noise, size=(n_quarter, 2))

    X1 = blob(np.array([+1.0, +1.0]))
    X2 = blob(np.array([+1.0, -1.0]))
    X3 = blob(np.array([-1.0, +1.0]))
    X4 = blob(np.array([-1.0, -1.0]))

    X = np.vstack([X1, X2, X3, X4])
    y = np.hstack([
        +np.ones(len(X1)),  # (+,+)
        -np.ones(len(X2)),  # (+,-)
        -np.ones(len(X3)),  # (-,+)
        +np.ones(len(X4)),  # (-,-)
    ])

    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_nearly_separable_with_noise(
    n_samples: int = 400,
    noise: float = 0.4,
    flip_y: float = 0.10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 2D dataset: two anisotropic Gaussian blobs with moderate overlap
    and label noise, to better differentiate perceptron variants.

    The positive and negative classes have different covariances (elliptical, rotated),
    making the optimal linear separator closer to a large-margin solution.
    """
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2

    # Class centers
    mean_pos = np.array([2.0, 0.0])
    mean_neg = np.array([-2.0, 0.0])

    # Anisotropic, slightly rotated covariances
    cov_pos = np.array([[1.2, 0.8], [0.8, 1.2]])
    cov_neg = np.array([[1.2, -0.8], [-0.8, 1.2]])

    X_pos = rng.multivariate_normal(mean_pos, cov_pos, size=n_half)
    X_neg = rng.multivariate_normal(mean_neg, cov_neg, size=n_samples - n_half)

    if noise > 0:
        X_pos += rng.normal(0, noise, size=X_pos.shape)
        X_neg += rng.normal(0, noise, size=X_neg.shape)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

    # Flip a fraction of labels to create outliers / mislabeled points
    n_flip = int(len(y) * flip_y)
    if n_flip > 0:
        idx = rng.choice(len(y), size=n_flip, replace=False)
        y[idx] *= -1.0

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_test_split_xy(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def make_gaussian_multiclass(
    n_classes: int = 6,
    n_samples_per_class: int = 80,
    n_features: int = 2,
    radius: float = 4.0,
    scale: float = 0.8,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a multi-class Gaussian dataset (2D or 3D) with N>5 classes.

    For 2D: class means are placed on a circle.
    For 3D: class means are placed on a sphere using evenly spaced azimuth angles
    and two elevation bands.

    Parameters
    ----------
    n_classes : int
        Number of classes (N>5 recommended).
    n_samples_per_class : int
        Samples per class.
    n_features : int
        Feature dimension (2 or 3 supported).
    radius : float
        Distance of class means from origin.
    scale : float
        Standard deviation for isotropic Gaussian per class.
    random_state : int
        RNG seed.

    Returns
    -------
    X : (n_samples, n_features) ndarray
    y : (n_samples,) ndarray of int labels in [0, n_classes-1]
    """
    assert n_features in (2, 3), "n_features must be 2 or 3"
    rng = np.random.default_rng(random_state)

    means = []
    if n_features == 2:
        angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
        for th in angles:
            means.append(np.array([radius * np.cos(th), radius * np.sin(th)]))
    else:
        # Distribute means over a sphere: two elevation bands
        n_band1 = n_classes // 2
        n_band2 = n_classes - n_band1
        az1 = np.linspace(0, 2 * np.pi, n_band1, endpoint=False)
        az2 = np.linspace(0, 2 * np.pi, n_band2, endpoint=False)
        el1 = np.deg2rad(35.0)
        el2 = np.deg2rad(70.0)
        for th in az1:
            means.append(np.array([
                radius * np.cos(th) * np.cos(el1),
                radius * np.sin(th) * np.cos(el1),
                radius * np.sin(el1),
            ]))
        for th in az2:
            means.append(np.array([
                radius * np.cos(th) * np.cos(el2),
                radius * np.sin(th) * np.cos(el2),
                radius * np.sin(el2),
            ]))

    X_list = []
    y_list = []
    cov = (scale ** 2) * np.eye(n_features)
    for i, mu in enumerate(means):
        Xi = rng.multivariate_normal(mean=mu, cov=cov, size=n_samples_per_class)
        yi = np.full(n_samples_per_class, i, dtype=int)
        X_list.append(Xi)
        y_list.append(yi)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]
