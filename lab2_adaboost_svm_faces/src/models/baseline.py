from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class BaselineResult:
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray
    confusion: np.ndarray
    pca_components: np.ndarray
    explained_variance_ratio: np.ndarray
    train_shape: Tuple[int, int]
    test_shape: Tuple[int, int]


def run_pca_svm_baseline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.3,
    random_state: int = 0,
    n_components: int = 120,
    kernel: Literal["linear", "poly", "rbf"] = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
) -> BaselineResult:
    """Train PCA + SVM baseline on flattened face pixels."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    svm = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=None, random_state=random_state)
    svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return BaselineResult(
        accuracy=float(acc),
        y_true=y_test,
        y_pred=y_pred,
        confusion=cm,
        pca_components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        train_shape=X_train_pca.shape,
        test_shape=X_test_pca.shape,
    )
