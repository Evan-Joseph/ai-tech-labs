from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_FILE = "pca_svm_recognizer.joblib"


@dataclass
class RecognitionResult:
    label: str
    confidence: float


class PCASVMRecognizer:
    """Encapsulate PCA + SVM face recognizer with persistence helpers."""

    def __init__(
        self,
        pca_components: int = 120,
        C: float = 10.0,
        gamma: str | float = "scale",
        random_state: int = 0,
    ) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("pca", PCA(n_components=pca_components, whiten=True, random_state=random_state)),
                ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=random_state)),
            ]
        )
        self.fitted = False
        self.label_encoder_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.pipeline.fit(X, y)
        classes = self.pipeline.named_steps["svm"].classes_
        self.label_encoder_ = np.array([f"ID {int(c):02d}" for c in classes])
        self.fitted = True

    def predict(self, X: np.ndarray) -> list[RecognitionResult]:
        if not self.fitted:
            raise RuntimeError("Recognizer is not fitted.")
        probabilities = self.pipeline.predict_proba(X)
        pred_indices = np.argmax(probabilities, axis=1)
        labels = self.label_encoder_[pred_indices]
        confidences = probabilities[np.arange(len(pred_indices)), pred_indices]
        return [RecognitionResult(label=label, confidence=float(conf)) for label, conf in zip(labels, confidences)]

    def save(self, path: Path) -> None:
        joblib.dump({
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder_,
            "fitted": self.fitted,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "PCASVMRecognizer":
        payload = joblib.load(path)
        recognizer = cls()
        recognizer.pipeline = payload["pipeline"]
        recognizer.label_encoder_ = payload["label_encoder"]
        recognizer.fitted = payload["fitted"]
        return recognizer
