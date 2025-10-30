from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np


def _to_label(y: np.ndarray) -> np.ndarray:
    """Ensure labels are in {-1, +1}. If labels are {0,1}, map 0->-1, 1->+1."""
    y = np.asarray(y).astype(float).ravel()
    uniq = np.unique(y)
    if set(uniq.tolist()) <= {0.0, 1.0}:
        y = np.where(y > 0, 1.0, -1.0)
    else:
        y = np.where(y > 0, 1.0, -1.0)
    return y


@dataclass
class TrainingHistory:
    errors_per_epoch: List[int] = field(default_factory=list)
    best_errors: Optional[int] = None
    epochs_ran: int = 0


class Perceptron:
    """Basic Perceptron classifier.

    Attributes
    ----------
    learning_rate: float
        Learning rate (eta) for updates.
    max_epochs: int
        Maximum number of passes over the training data.
    shuffle: bool
        Whether to shuffle samples every epoch.
    random_state: Optional[int]
        Random seed for reproducibility.

    Notes
    -----
    Update rule for a misclassified sample (x, y):
        w <- w + eta * y * x
        b <- b + eta * y
    where y in {-1, +1}.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_epochs: int = 1000,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.history_ = TrainingHistory()
        self.is_fitted_: bool = False

    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=1e-3, size=n_features)
        self.b_ = 0.0

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.w_ is None:
            raise RuntimeError("Model is not fitted: weights are None")
        return X @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1.0, -1.0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = _to_label(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Fit the perceptron on training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) with labels in {-1, +1} or {0, 1}

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = _to_label(y)
        n_samples, n_features = X.shape

        self._init_params(n_features)
        rng = np.random.default_rng(self.random_state)

        self.history_ = TrainingHistory(errors_per_epoch=[], best_errors=None, epochs_ran=0)

        for epoch in range(self.max_epochs):
            if self.shuffle:
                idx = rng.permutation(n_samples)
                X_epoch = X[idx]
                y_epoch = y[idx]
            else:
                X_epoch = X
                y_epoch = y

            errors = 0
            for xi, yi in zip(X_epoch, y_epoch):
                margin = yi * (np.dot(xi, self.w_) + self.b_)
                if margin <= 0:
                    # misclassified, update
                    self.w_ = self.w_ + self.learning_rate * yi * xi
                    self.b_ = self.b_ + self.learning_rate * yi
                    errors += 1

            self.history_.errors_per_epoch.append(int(errors))
            self.history_.epochs_ran = epoch + 1

            if errors == 0:
                break  # converged on linearly separable data

        self.is_fitted_ = True
        return self


class PocketPerceptron(Perceptron):
    """Perceptron with the Pocket algorithm.

    Keeps the best weights (with minimal training errors) observed during training.
    Useful when data are not strictly linearly separable.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_epochs: int = 1000,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(learning_rate, max_epochs, shuffle, random_state)
        self._pocket_w: Optional[np.ndarray] = None
        self._pocket_b: float = 0.0
        self._pocket_errors: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PocketPerceptron":
        X = np.asarray(X, dtype=float)
        y = _to_label(y)
        n_samples, n_features = X.shape

        self._init_params(n_features)
        rng = np.random.default_rng(self.random_state)

        self._pocket_w = self.w_.copy()
        self._pocket_b = float(self.b_)

        # initial errors
        initial_pred = np.where(X @ self.w_ + self.b_ >= 0, 1.0, -1.0)
        initial_errors = int(np.sum(initial_pred != y))
        self._pocket_errors = initial_errors
        self.history_ = TrainingHistory(errors_per_epoch=[], best_errors=initial_errors, epochs_ran=0)

        for epoch in range(self.max_epochs):
            if self.shuffle:
                idx = rng.permutation(n_samples)
                X_epoch = X[idx]
                y_epoch = y[idx]
            else:
                X_epoch = X
                y_epoch = y

            errors = 0
            for xi, yi in zip(X_epoch, y_epoch):
                margin = yi * (np.dot(xi, self.w_) + self.b_)
                if margin <= 0:
                    self.w_ = self.w_ + self.learning_rate * yi * xi
                    self.b_ = self.b_ + self.learning_rate * yi

            # measure errors after this epoch
            pred = np.where(X @ self.w_ + self.b_ >= 0, 1.0, -1.0)
            errors = int(np.sum(pred != y))
            self.history_.errors_per_epoch.append(errors)
            self.history_.epochs_ran = epoch + 1

            # update pocket if better
            if errors < self._pocket_errors:
                self._pocket_errors = errors
                self._pocket_w = self.w_.copy()
                self._pocket_b = float(self.b_)
                self.history_.best_errors = errors

            # early stop if perfect
            if errors == 0:
                break

        # use pocket weights for inference
        self.w_ = self._pocket_w
        self.b_ = self._pocket_b
        self.is_fitted_ = True
        return self


class AveragedPerceptron(Perceptron):
    """Averaged Perceptron.

    Maintains an average of weights across all updates to improve generalization.

    Reference: Freund and Schapire (1999) - Large Margin Classification Using the Perceptron Algorithm.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_epochs: int = 20,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(learning_rate, max_epochs, shuffle, random_state)
        self.w_avg_: Optional[np.ndarray] = None
        self.b_avg_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AveragedPerceptron":
        X = np.asarray(X, dtype=float)
        y = _to_label(y)
        n_samples, n_features = X.shape

        self._init_params(n_features)
        rng = np.random.default_rng(self.random_state)

        w = self.w_.copy()
        b = float(self.b_)
        w_sum = np.zeros_like(w)
        b_sum = 0.0
        counter = 0

        self.history_ = TrainingHistory(errors_per_epoch=[], best_errors=None, epochs_ran=0)

        for epoch in range(self.max_epochs):
            if self.shuffle:
                idx = rng.permutation(n_samples)
                X_epoch = X[idx]
                y_epoch = y[idx]
            else:
                X_epoch = X
                y_epoch = y

            errors = 0
            for xi, yi in zip(X_epoch, y_epoch):
                if yi * (np.dot(xi, w) + b) <= 0:
                    w = w + self.learning_rate * yi * xi
                    b = b + self.learning_rate * yi
                    errors += 1
                # accumulate regardless
                w_sum += w
                b_sum += b
                counter += 1

            self.history_.errors_per_epoch.append(int(errors))
            self.history_.epochs_ran = epoch + 1

        # averaged parameters
        self.w_ = w
        self.b_ = b
        if counter > 0:
            self.w_avg_ = w_sum / counter
            self.b_avg_ = b_sum / counter
        else:
            self.w_avg_ = w.copy()
            self.b_avg_ = float(b)

        self.is_fitted_ = True
        return self

    def decision_function(self, X: np.ndarray, use_average: bool = True) -> np.ndarray:
        X = np.asarray(X)
        if use_average and self.w_avg_ is not None:
            return X @ self.w_avg_ + self.b_avg_
        return super().decision_function(X)

    def predict(self, X: np.ndarray, use_average: bool = True) -> np.ndarray:
        scores = self.decision_function(X, use_average=use_average)
        return np.where(scores >= 0, 1.0, -1.0)


class MarginPerceptron(Perceptron):
    """Perceptron with margin.

    Updates when the functional margin is below a threshold `margin`.

    Parameters
    ----------
    margin : float
        Desired margin; if y*(w·x+b) <= margin, perform an update.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_epochs: int = 1000,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
        margin: float = 1.0,
    ) -> None:
        super().__init__(learning_rate, max_epochs, shuffle, random_state)
        self.margin = float(margin)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MarginPerceptron":
        X = np.asarray(X, dtype=float)
        y = _to_label(y)
        n_samples, n_features = X.shape

        self._init_params(n_features)
        rng = np.random.default_rng(self.random_state)

        self.history_ = TrainingHistory(errors_per_epoch=[], best_errors=None, epochs_ran=0)

        for epoch in range(self.max_epochs):
            if self.shuffle:
                idx = rng.permutation(n_samples)
                X_epoch = X[idx]
                y_epoch = y[idx]
            else:
                X_epoch = X
                y_epoch = y

            errors = 0
            for xi, yi in zip(X_epoch, y_epoch):
                margin_val = yi * (np.dot(xi, self.w_) + self.b_)
                if margin_val <= self.margin:
                    self.w_ = self.w_ + self.learning_rate * yi * xi
                    self.b_ = self.b_ + self.learning_rate * yi
                    # Consider update as an error if actually misclassified
                    if margin_val <= 0:
                        errors += 1

            self.history_.errors_per_epoch.append(int(errors))
            self.history_.epochs_ran = epoch + 1

        self.is_fitted_ = True
        return self
