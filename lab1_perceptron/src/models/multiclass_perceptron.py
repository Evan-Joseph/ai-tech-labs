from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class TrainingHistoryMC:
    errors_per_epoch: List[int] = field(default_factory=list)
    epochs_ran: int = 0


class MultiClassPerceptronMax:
    """Kesler 构造的多类感知器（最大值判决）。

    每一类 i 对应一个判别函数 g_i(x) = w_i^T x + b_i，预测为 argmax_i g_i(x)。
    当真实类 y 的得分不大于其他任何一类的得分（或未满足带余量约束）时，执行：
        w_y += eta * x,   b_y += eta
        w_t -= eta * x,   b_t -= eta      其中 t = argmax_{k != y} g_k(x)

    Parameters
    ----------
    learning_rate : float
        学习率。
    max_epochs : int
        最大训练轮数。
    shuffle : bool
        是否每轮打乱样本。
    random_state : Optional[int]
        随机种子。
    margin : float
        余量阈值 b，当 g_y(x) - max_{t!=y} g_t(x) < b 时触发更新（b=0 退化为标准规则）。
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_epochs: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
        margin: float = 0.0,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self.margin = float(margin)

        self.W_: Optional[np.ndarray] = None  # shape (K, D)
        self.b_: Optional[np.ndarray] = None  # shape (K,)
        self.is_fitted_: bool = False
        self.classes_: Optional[np.ndarray] = None
        self.history_ = TrainingHistoryMC()

    def _init_params(self, n_classes: int, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(loc=0.0, scale=1e-3, size=(n_classes, n_features))
        self.b_ = np.zeros(n_classes, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiClassPerceptronMax":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        n_classes = len(classes)
        # map labels to indices 0..K-1 if needed
        label_to_index = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([label_to_index[yy] for yy in y], dtype=int)

        self._init_params(n_classes, n_features)
        rng = np.random.default_rng(self.random_state)
        self.history_ = TrainingHistoryMC(errors_per_epoch=[], epochs_ran=0)

        for epoch in range(self.max_epochs):
            if self.shuffle:
                idx = rng.permutation(n_samples)
                X_epoch, y_epoch = X[idx], y_idx[idx]
            else:
                X_epoch, y_epoch = X, y_idx

            errors = 0
            for xi, yi in zip(X_epoch, y_epoch):
                scores = self.W_ @ xi + self.b_
                # best competing class t != yi
                t = int(np.argmax(scores))
                # margin violation if yi not the argmax or margin gap < self.margin
                if (t != yi) or (scores[yi] - np.max(np.delete(scores, yi)) < self.margin):
                    # find the top competitor excluding yi
                    t_star = int(np.argmax(np.delete(scores, yi)))
                    # np.delete changed index; need the original class index
                    # recover t_star index in original space
                    if t_star >= yi:
                        t_star += 1
                    # update
                    self.W_[yi] += self.learning_rate * xi
                    self.b_[yi] += self.learning_rate
                    self.W_[t_star] -= self.learning_rate * xi
                    self.b_[t_star] -= self.learning_rate
                    errors += 1

            self.history_.errors_per_epoch.append(int(errors))
            self.history_.epochs_ran = epoch + 1
            if errors == 0:
                break

        self.is_fitted_ = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=float)
        return X @ self.W_.T + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        # map back to original class labels
        return self.classes_[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
