from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.datasets import fetch_olivetti_faces


@dataclass(frozen=True)
class ORLFaces:
    """Container for ORL (Olivetti) faces dataset."""

    data: np.ndarray
    images: np.ndarray
    target: np.ndarray
    target_names: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.data.shape[1])

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(self.images.shape[1:3])  # type: ignore[return-value]


def load_orl_faces(
    data_home: Optional[Path | str] = None,
    *,
    shuffle: bool = True,
    random_state: int = 0,
    download_if_missing: bool = True,
) -> ORLFaces:
    """Fetch the ORL (Olivetti) faces dataset via scikit-learn."""

    dataset = fetch_olivetti_faces(
        data_home=None if data_home is None else str(data_home),
        shuffle=shuffle,
        random_state=random_state,
        download_if_missing=download_if_missing,
    )
    target = dataset.target
    if hasattr(dataset, "target_names") and dataset.target_names is not None:
        target_names = np.asarray(dataset.target_names)
    else:
        unique_ids = np.unique(target)
        target_names = np.array([f"ID {idx:02d}" for idx in unique_ids])

    return ORLFaces(
        data=dataset.data,
        images=dataset.images,
        target=target,
        target_names=target_names,
    )
